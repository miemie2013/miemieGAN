import cv2
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import math
import contextlib

from mmgan.models.initializer import kaiming_normal_, constant_
from mmgan.utils.visual import tensor2img, make_grid


@contextlib.contextmanager
def ddp_sync(module, sync):
    if sync or not isinstance(module, torch.nn.parallel.DistributedDataParallel):
        yield
    else:
        with module.no_sync():
            yield


weight_gradients_disabled = False   # Forcefully disable computation of gradients with respect to the weights.

@contextlib.contextmanager
def no_weight_gradients():
    global weight_gradients_disabled
    old = weight_gradients_disabled
    weight_gradients_disabled = True
    yield
    weight_gradients_disabled = old


def save_tensor(dic, key, tensor):
    if tensor is not None:  # 有的梯度张量可能是None
        dic[key] = tensor.cpu().detach().numpy()

def print_diff(dic, key, tensor):
    if tensor is not None:  # 有的梯度张量可能是None
        ddd = np.sum((dic[key] - tensor.cpu().detach().numpy()) ** 2)
        print('diff=%.6f (%s)' % (ddd, key))



def translate_using_reference(nets, w_hpf, x_src, x_ref, y_ref):
    N, C, H, W = x_src.shape
    wb = torch.Tensor(np.ones((1, C, H, W))).to(torch.float32)
    x_src_with_wb = torch.cat([wb, x_src], 0)

    masks = nets['fan'].get_heatmap(x_src) if w_hpf > 0 else None
    s_ref = nets['style_encoder'](x_ref, y_ref)
    s_ref_list = s_ref.unsqueeze(1)
    s_ref_lists = []
    for _ in range(N):
        s_ref_lists.append(s_ref_list)
    s_ref_list = torch.stack(s_ref_lists, 1)
    s_ref_list = torch.reshape(
        s_ref_list,
        (s_ref_list.shape[0], s_ref_list.shape[1], s_ref_list.shape[3]))
    x_concat = [x_src_with_wb]
    for i, s_ref in enumerate(s_ref_list):
        x_fake = nets['generator'](x_src, s_ref, masks=masks)
        x_fake_with_ref = torch.cat([x_ref[i:i + 1], x_fake], 0)
        x_concat += [x_fake_with_ref]

    x_concat = torch.cat(x_concat, 0)
    img = tensor2img(make_grid(x_concat, nrow=N + 1, range=(0, 1)))
    del x_concat
    return img


def compute_d_loss(nets,
                   lambda_reg,
                   x_real,
                   y_org,
                   y_trg,
                   z_trg=None,
                   x_ref=None,
                   masks=None):
    assert (z_trg is None) != (x_ref is None)
    # with real images
    x_real.requires_grad = True   # x_real [N, 3, 256, 256]
    out = nets['discriminator'](x_real, y_org)  # [N, ]  输出只取真实类别处的输出。
    loss_real = adv_loss(out, 1)  # [N, ]  交叉熵损失。这是真的图像
    loss_reg = r1_reg(out, x_real)  # [N, ]  梯度惩罚损失

    # with fake images
    with torch.no_grad():  # 训练判别器时，生成器前向传播应停止梯度。
        if z_trg is not None:
            s_trg = nets['mapping_network'](z_trg, y_trg)   # (N, style_dim)  随机噪声z_trg生成风格编码s_trg，只取目标domain的输出
        else:  # x_ref is not None
            s_trg = nets['style_encoder'](x_ref, y_trg)   # (N, style_dim)  目标domain真实图像x_ref生成风格编码s_trg

        x_fake = nets['generator'](x_real, s_trg, masks=masks)  # 风格编码s_trg和真实图像生成目标domain的图像x_fake
    out = nets['discriminator'](x_fake, y_trg)  # x_fake [N, 3, 256, 256]  注意，x_fake已经停止梯度。   out [N, ]  输出只取真实(目标domain)类别处的输出。
    loss_fake = adv_loss(out, 0)  # [N, ]  交叉熵损失。这是假的图像

    loss = loss_real + loss_fake + lambda_reg * loss_reg   # lambda_reg是梯度惩罚损失的权重
    return loss, {
        'real': loss_real.cpu().detach().numpy(),
        'fake': loss_fake.cpu().detach().numpy(),
        'reg': loss_reg.cpu().detach().numpy()
    }


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)  # [N, ]  标记
    loss = F.binary_cross_entropy_with_logits(logits, targets)  # [N, ]  交叉熵损失
    return loss


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.shape[0]
    grad_dout = torch.autograd.grad(outputs=[d_out.sum()], inputs=[x_in], create_graph=True, only_inputs=True)[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.shape == x_in.shape)
    reg = 0.5 * torch.reshape(grad_dout2, (batch_size, -1)).sum(1).mean(0)
    return reg


def dump_model(model):
    params = {}
    for k in model.state_dict().keys():
        if k.endswith('.scale'):
            params[k] = model.state_dict()[k].shape
    return params


def compute_g_loss(nets,
                   w_hpf,
                   lambda_sty,
                   lambda_ds,
                   lambda_cyc,
                   x_real,
                   y_org,
                   y_trg,
                   z_trgs=None,
                   x_refs=None,
                   masks=None):
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs

    # adversarial loss。对抗损失
    if z_trgs is not None:
        s_trg = nets['mapping_network'](z_trg, y_trg)   # 第1个随机噪声z_trg 生成风格编码s_trg
    else:
        s_trg = nets['style_encoder'](x_ref, y_trg)   # 目标domain的第1张真实图像x_ref 生成风格编码s_trg

    x_fake = nets['generator'](x_real, s_trg, masks=masks)   # 真实图像和第1个风格编码s_trg生成第1张假图像
    # 这里不先把discriminator冻结起来吗？懂了，后面没有optimizers['discriminator'].step()这句代码，所以discriminator的参数不会更新的。
    out = nets['discriminator'](x_fake, y_trg)
    loss_adv = adv_loss(out, 1)

    # style reconstruction loss。风格重构损失
    s_pred = nets['style_encoder'](x_fake, y_trg)   # (N, style_dim)  假图像生成对应domain的风格编码s_pred
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))   # 假图像生成对应domain的风格编码s_pred 和 s_trg 取绝对值损失。

    # diversity sensitive loss。差异敏感损失
    if z_trgs is not None:
        s_trg2 = nets['mapping_network'](z_trg2, y_trg)   # 第2个随机噪声z_trg2 生成风格编码s_trg2
    else:
        s_trg2 = nets['style_encoder'](x_ref2, y_trg)   # 目标domain的第2张真实图像x_ref2 生成风格编码s_trg2
    x_fake2 = nets['generator'](x_real, s_trg2, masks=masks)   # 真实图像和第2个风格编码s_trg2生成第2张假图像
    loss_ds = torch.mean(torch.abs(x_fake - x_fake2))   # 第1张假图像 和 第2张假图像 取绝对值损失。

    # cycle-consistency loss。循环一致性损失
    if w_hpf > 0:
        if isinstance(nets['fan'], torch.nn.parallel.DistributedDataParallel):
            masks = nets['fan'].module.get_heatmap(x_fake)
        else:
            masks = nets['fan'].get_heatmap(x_fake)
    else:
        masks = None

    s_org = nets['style_encoder'](x_real, y_org)   # x_real 生成风格编码s_org
    x_rec = nets['generator'](x_fake, s_org, masks=masks)   # x_fake“变回”x_real(x_rec)
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))   # x_real 和 x_rec 取绝对值损失。

    loss = loss_adv + lambda_sty * loss_sty \
        - lambda_ds * loss_ds + lambda_cyc * loss_cyc
    return loss, {
        'adv': loss_adv.cpu().detach().numpy(),
        'sty': loss_sty.cpu().detach().numpy(),
        'ds:': loss_ds.cpu().detach().numpy(),
        'cyc': loss_cyc.cpu().detach().numpy()
    }


def soft_update(source, ema_model, beta=1.0):
    '''
    ema:
    ema = beta * ema + (1. - beta) * source

    '''
    assert 0.0 <= beta <= 1.0
    for p_ema, p in zip(ema_model.parameters(), source.parameters()):
        p_ema.copy_(p.lerp(p_ema, beta))  # p_ema = beta * p_ema + (1 - beta) * p
    for b_ema, b in zip(ema_model.buffers(), source.buffers()):
        b_ema.copy_(b)


def he_init(module):
    if isinstance(module, nn.Conv2d):
        kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            constant_(module.bias, 0)


class StarGANv2Model:
    def __init__(
        self,
        generator,
        generator_ema,
        style,
        style_ema,
        mapping,
        mapping_ema,
        fan=None,
        fan_ema=None,
        discriminator=None,
        device=None,
        rank=0,
        latent_dim=16,
        lambda_reg=1,
        lambda_sty=1,
        lambda_ds=1,
        lambda_cyc=1,
    ):
        super(StarGANv2Model, self).__init__()
        self.optimizers = OrderedDict()
        self.metrics = OrderedDict()
        self.losses = OrderedDict()
        self.visual_items = OrderedDict()
        self.nets = OrderedDict()
        self.nets_ema = OrderedDict()

        self.generator = generator
        self.generator_ema = generator_ema
        self.style = style
        self.style_ema = style_ema
        self.mapping = mapping
        self.mapping_ema = mapping_ema
        self.fan = fan
        self.fan_ema = fan_ema
        self.generator.train()
        self.style.train()
        self.mapping.train()
        self.generator_ema.eval()
        self.style_ema.eval()
        self.mapping_ema.eval()
        if fan is not None:
            self.fan.eval()
            self.fan_ema.eval()
        if discriminator:
            self.discriminator = discriminator
            self.discriminator.train()
        self.w_hpf = self.generator.w_hpf
        self.latent_dim = latent_dim
        self.lambda_reg = lambda_reg
        self.lambda_sty = lambda_sty
        self.lambda_ds = lambda_ds
        self.lambda_cyc = lambda_cyc

        self.device = device
        self.rank = rank

        self.generator.apply(he_init)
        self.style.apply(he_init)
        self.mapping.apply(he_init)
        self.discriminator.apply(he_init)

        # remember the initial value of ds weight
        # 记住最初的lambda_ds
        self.initial_lambda_ds = self.lambda_ds

        # G和G_ema有相同的初始权重
        soft_update(self.mapping, self.mapping_ema, beta=0.0)
        soft_update(self.generator, self.generator_ema, beta=0.0)
        soft_update(self.style, self.style_ema, beta=0.0)
        self.nets['generator'] = self.generator
        self.nets['style_encoder'] = self.style
        self.nets['mapping_network'] = self.mapping
        self.nets['discriminator'] = self.discriminator
        self.nets['fan'] = self.fan
        self.nets_ema['generator'] = self.generator_ema
        self.nets_ema['style_encoder'] = self.style_ema
        self.nets_ema['mapping_network'] = self.mapping_ema
        self.nets_ema['fan'] = self.fan_ema


    def setup_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Args:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.input = input
        self.input['z_trg'] = torch.randn((input['src'].shape[0], self.latent_dim))
        self.input['z_trg2'] = torch.randn((input['src'].shape[0], self.latent_dim))

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    def _reset_grad(self, optims):
        for optim in optims.values():
            optim.zero_grad(set_to_none=True)

    def train_iter(self, optimizers=None):
        # x_real [N, 3, 256, 256]
        # y_org  [N, ]  x_real的类别id
        x_real, y_org = self.input['src'], self.input['src_cls']
        # x_ref  [N, 3, 256, 256]
        # x_ref2 [N, 3, 256, 256]  x_real的类别id
        # y_trg  [N, ]  x_ref和x_ref2的类别id
        x_ref, x_ref2, y_trg = self.input['ref'], self.input['ref2'], self.input['ref_cls']
        # z_trg  [N, 16]  随机噪声z
        # z_trg2 [N, 16]  随机噪声z2
        z_trg, z_trg2 = self.input['z_trg'], self.input['z_trg2']

        if self.w_hpf > 0:
            if isinstance(self.nets['fan'], torch.nn.parallel.DistributedDataParallel):
                masks = self.nets['fan'].modeule.get_heatmap(x_real)
            else:
                masks = self.nets['fan'].get_heatmap(x_real)
        else:
            masks = None

        # 查看masks
        # m0, m1 = masks
        # aaa = x_real.numpy()[0]
        # aaa = aaa.transpose(1, 2, 0)
        # aaa = (aaa + 1.0) * 127.5
        # aaa = cv2.cvtColor(aaa, cv2.COLOR_RGB2BGR)
        # cv2.imwrite('aaa1.png', aaa)
        # m0 = m0.numpy()[0][0]
        # m1 = m1.numpy()[0][0]
        # cv2.imwrite('aaa2.png', m0 * 255.0)
        # cv2.imwrite('aaa3.png', m1 * 255.0)

        # ================ train the discriminator ================
        # 训练了2次判别器。第1次和第2次的区别是如何生成假图像：
        # 第1次用      随机噪声z_trg     经过mapping_network生成 风格编码s_trg，再用s_trg和真实图像x_real生成假图像；
        # 第2次用 目标domain真实图像x_ref  经过style_encoder生成  风格编码s_trg，再用s_trg和真实图像x_real生成假图像；

        # lambda_reg是梯度惩罚损失的权重。包括计算
        # (1)真实图像的判断损失（交叉熵）；(2)真实图像的梯度惩罚损失；
        # (3)随机噪声z_trg和真实图像x_real生成的假图像的判断损失（交叉熵）；
        d_loss, d_losses_latent = compute_d_loss(self.nets,
                                                 self.lambda_reg,
                                                 x_real,
                                                 y_org,
                                                 y_trg,
                                                 z_trg=z_trg,
                                                 masks=masks)
        self._reset_grad(optimizers)  # 梯度清0
        d_loss.backward()  # 反向传播
        optimizers['discriminator'].minimize(d_loss)  # 更新参数

        # lambda_reg是梯度惩罚损失的权重。包括计算
        # (1)真实图像的判断损失（交叉熵）；(2)真实图像的梯度惩罚损失；
        # (3)目标domain真实图像x_ref和真实图像x_real生成的假图像的判断损失（交叉熵）；
        d_loss, d_losses_ref = compute_d_loss(self.nets,
                                              self.lambda_reg,
                                              x_real,
                                              y_org,
                                              y_trg,
                                              x_ref=x_ref,
                                              masks=masks)
        self._reset_grad(optimizers)  # 梯度清0
        d_loss.backward()  # 反向传播
        optimizers['discriminator'].step()  # 更新参数

        # ================ train the generator ================
        # 训练了2次生成器。第1次和第2次的区别是如何生成假图像：
        # 第1次用      随机噪声z_trg     经过mapping_network生成 风格编码s_trg，再用s_trg和真实图像x_real生成假图像；
        # 第2次用 目标domain真实图像x_ref  经过style_encoder生成  风格编码s_trg，再用s_trg和真实图像x_real生成假图像；

        g_loss, g_losses_latent = compute_g_loss(self.nets,
                                                 self.w_hpf,
                                                 self.lambda_sty,
                                                 self.lambda_ds,
                                                 self.lambda_cyc,
                                                 x_real,
                                                 y_org,
                                                 y_trg,
                                                 z_trgs=[z_trg, z_trg2],
                                                 masks=masks)
        self._reset_grad(optimizers)
        g_loss.backward()
        optimizers['generator'].step()
        optimizers['mapping_network'].step()
        optimizers['style_encoder'].step()

        g_loss, g_losses_ref = compute_g_loss(self.nets,
                                              self.w_hpf,
                                              self.lambda_sty,
                                              self.lambda_ds,
                                              self.lambda_cyc,
                                              x_real,
                                              y_org,
                                              y_trg,
                                              x_refs=[x_ref, x_ref2],
                                              masks=masks)
        self._reset_grad(optimizers)
        g_loss.backward()
        optimizers['generator'].step()

        # compute moving average of network parameters。指数滑动平均
        soft_update(self.mapping, self.mapping_ema, beta=0.999)
        soft_update(self.generator, self.generator_ema, beta=0.999)
        soft_update(self.style, self.style_ema, beta=0.999)

        # decay weight for diversity sensitive loss
        if self.lambda_ds > 0:
            self.lambda_ds -= (self.initial_lambda_ds / self.total_iter)

        for loss, prefix in zip(
            [d_losses_latent, d_losses_ref, g_losses_latent, g_losses_ref],
            ['D/latent_', 'D/ref_', 'G/latent_', 'G/ref_']):
            for key, value in loss.items():
                self.losses[prefix + key] = value
        self.losses['G/lambda_ds'] = self.lambda_ds
        self.losses['Total iter'] = int(self.total_iter)

        return loss_numpys

    @torch.no_grad()
    def test_iter(self, metrics=None):
        self.generator_ema.eval()
        self.style_ema.eval()
        soft_update(self.mapping, self.mapping_ema, beta=0.999)
        soft_update(self.generator, self.generator_ema, beta=0.999)
        soft_update(self.style, self.style_ema, beta=0.999)
        src_img = self.input['src']
        ref_img = self.input['ref']
        ref_label = self.input['ref_cls']
        with torch.no_grad():
            img = translate_using_reference(
                self.nets_ema, self.w_hpf,
                paddle.to_tensor(src_img).astype('float32'),
                paddle.to_tensor(ref_img).astype('float32'),
                paddle.to_tensor(ref_label).astype('float32'))
        return img_bgr, seed
