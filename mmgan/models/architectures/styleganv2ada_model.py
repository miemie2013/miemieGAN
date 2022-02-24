import cv2
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys


def soft_update(source, target, beta=1.0):
    '''
    ema:
    target = beta * source + (1. - beta) * target

    '''
    assert 0.0 <= beta <= 1.0

    if isinstance(source, paddle.DataParallel):
        source = source._layers

    target_model_map = dict(target.named_parameters())
    for param_name, source_param in source.named_parameters():
        target_param = target_model_map[param_name]
        target_param.set_value(beta * source_param +
                               (1.0 - beta) * target_param)


def dump_model(model):
    params = {}
    for k in model.state_dict().keys():
        if k.endswith('.scale'):
            params[k] = model.state_dict()[k].shape
    return params



def he_init(module):
    if isinstance(module, nn.Conv2D):
        kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            constant_(module.bias, 0)


class StyleGANv2ADAModel(torch.nn.Module):
    def __init__(
        self,
        synthesis,
        synthesis_ema,
        mapping,
        mapping_ema,
        discriminator=None,
        G_reg_interval=4,
        D_reg_interval=16,
        augment_pipe=None,
        style_mixing_prob=0.9,
        r1_gamma=10,
        pl_batch_shrink=2,
        pl_decay=0.01,
        pl_weight=2.0,
    ):
        super(StyleGANv2ADAModel, self).__init__()
        self.optimizers = OrderedDict()
        self.metrics = OrderedDict()
        self.losses = OrderedDict()
        self.visual_items = OrderedDict()

        self.synthesis = synthesis
        self.synthesis_ema = synthesis_ema
        self.mapping = mapping
        self.mapping_ema = mapping_ema
        if discriminator:
            self.discriminator = discriminator
        self.c_dim = mapping.c_dim
        self.z_dim = mapping.z_dim
        self.w_dim = mapping.w_dim

        self.phases = []
        for name, reg_interval in [('G', G_reg_interval), ('D', D_reg_interval)]:
            if reg_interval is None:
                # opt = dnnlib.util.construct_class_by_name(params=module.parameters(),
                #                                           **opt_kwargs)  # subclass of torch.optim.Optimizer
                # phases += [dnnlib.EasyDict(name=name + 'both', module=module, opt=opt, interval=1)]
                pass
            else:  # Lazy regularization.
                self.phases += [dict(name=name + 'main', interval=1)]
                self.phases += [dict(name=name + 'reg', interval=reg_interval)]

        self.z_dim = self.mapping.z_dim
        self.batch_idx = 0

        # loss config.
        # self.augment_pipe = build_generator(augment_pipe)
        self.augment_pipe = None
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight

        self.pl_mean = None



    def setup_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Args:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.input = input

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    def _reset_grad(self, optims):
        for optim in optims.values():
            optim.zero_grad()

    def run_G(self, z, c, sync):
        # print('------------------ run_G -------------------')
        ws = self.mapping(z, c)
        # self.style_mixing_prob = -1.0
        if self.style_mixing_prob > 0:
            cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
            cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
            ws[:, cutoff:] = self.mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]

        img = self.synthesis(ws)
        return img, ws

    def run_D(self, img, c, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        logits = self.discriminator(img, c)
        return logits

    # 梯度累加（变相增大批大小）。dic2是为了梯度对齐。
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
    # def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain, dic2=None):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        loss_numpy = {}

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl)) # May get synced by Gpl.
            # aaaaaaaaa1 = dic2[phase + 'gen_img']
            # aaaaaaaaa2 = gen_img.cpu().detach().numpy()
            # ddd = np.mean((dic2[phase + 'gen_img'] - gen_img.cpu().detach().numpy()) ** 2)
            # print('ddd=%.6f' % ddd)
            # ddd = np.mean((dic2[phase + '_gen_ws'] - _gen_ws.cpu().detach().numpy()) ** 2)
            # print('ddd=%.6f' % ddd)

            gen_logits = self.run_D(gen_img, gen_c, sync=False)
            # ddd = np.mean((dic2[phase + 'gen_logits'] - gen_logits.cpu().detach().numpy()) ** 2)
            # print('ddd=%.6f' % ddd)

            loss_Gmain = torch.nn.functional.softplus(-gen_logits)  # -log(sigmoid(gen_logits))
            # loss_Gmain = loss_Gmain.mean()
            # loss_numpy['loss_Gmain'] = loss_Gmain.cpu().detach().numpy()

            # loss_G = loss_Gmain
            # loss_G = loss_G * float(gain)
            # loss_G.backward()  # 咩酱：gain即上文提到的这个阶段的训练间隔。
            loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            # print('----------------- do_Gpl -----------------')
            batch_size = gen_z.shape[0] // self.pl_batch_shrink
            # with misc.ddp_sync(self.G_flownet, sync):
            #     flow = self.G_flownet(torch.cat((cloth[:batch_size], aff_pose[:batch_size]), dim=1))
            # warp_cloth = F.grid_sample(cloth[:batch_size, :3, :, :], flow)

            gen_c_ = None
            if gen_c is not None:
                gen_c_ = gen_c[:batch_size]

            gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c_, sync=sync)
            # ddd = np.mean((dic2[phase + 'gen_img'] - gen_img.cpu().detach().numpy()) ** 2)
            # print('ddd=%.6f' % ddd)
            # ddd = np.mean((dic2[phase + 'gen_ws'] - gen_ws.cpu().detach().numpy()) ** 2)
            # print('ddd=%.6f' % ddd)
            pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
            # pl_noise = torch.ones_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
            pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]

            pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
            # ddd = np.mean((dic2[phase + 'pl_grads'] - pl_grads.cpu().detach().numpy()) ** 2)
            # print('ddd=%.6f' % ddd)
            # ddd = np.mean((dic2[phase + 'pl_lengths'] - pl_lengths.cpu().detach().numpy()) ** 2)
            # print('ddd=%.6f' % ddd)
            if self.pl_mean is None:
                self.pl_mean = torch.zeros([1, ], dtype=torch.float32, device=pl_lengths.device)
            pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
            self.pl_mean.copy_(pl_mean.detach())

            pl_penalty = (pl_lengths - pl_mean).square()
            loss_Gpl = pl_penalty * self.pl_weight

            # loss_Gpl = (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean() * float(gain)
            # loss_numpy['loss_Gpl'] = loss_Gpl.cpu().detach().numpy()
            # loss_Gpl.backward()  # 咩酱：gain即上文提到的这个阶段的训练间隔。
            (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        # loss3 = 0.0
        if do_Dmain:
            gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=False)
            # ddd = np.mean((dic2[phase + 'gen_img'] - gen_img.cpu().detach().numpy()) ** 2)
            # print('ddd=%.6f' % ddd)
            # ddd = np.mean((dic2[phase + '_gen_ws'] - _gen_ws.cpu().detach().numpy()) ** 2)
            # print('ddd=%.6f' % ddd)
            gen_logits = self.run_D(gen_img, gen_c, sync=False) # Gets synced by loss_Dreal.
            # ddd = np.mean((dic2[phase + 'gen_logits'] - gen_logits.cpu().detach().numpy()) ** 2)
            # print('ddd=%.6f' % ddd)

            loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            loss_Dgen.mean().mul(gain).backward()
            # loss_Dgen = loss_Dgen.mean()
            # loss_numpy['loss_Dgen'] = loss_Dgen.cpu().detach().numpy()

            # loss3 = loss_Dgen * float(gain)

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'

            real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
            real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
            # ddd = np.mean((dic2[phase + 'real_logits'] - real_logits.cpu().detach().numpy()) ** 2)
            # print('ddd=%.6f' % ddd)

            loss_Dreal = 0
            if do_Dmain:
                loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                # ddd = np.mean((dic2[phase + 'loss_Dreal'] - loss_Dreal.cpu().detach().numpy()) ** 2)
                # print('ddd=%.6f' % ddd)
                # loss_numpy['loss_Dreal'] = loss_Dreal.cpu().detach().numpy().mean()

            loss_Dr1 = 0
            if do_Dr1:
                r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                # ddd = np.mean((dic2[phase + 'r1_grads'] - r1_grads.cpu().detach().numpy()) ** 2)
                # print('ddd=%.6f' % ddd)
                r1_penalty = r1_grads.square().sum([1, 2, 3])
                # ddd = np.mean((dic2[phase + 'r1_penalty'] - r1_penalty.cpu().detach().numpy()) ** 2)
                # print('ddd=%.6f' % ddd)
                loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                # loss_numpy['loss_Dr1'] = loss_Dr1.cpu().detach().numpy().mean()

            # loss4 = (loss_Dreal + loss_Dr1).mean() * float(gain)
            # if do_Dmain:
            #     loss4 += loss3
            # loss4.backward()  # 咩酱：gain即上文提到的这个阶段的训练间隔。
            (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()
        return loss_numpy

    def train_iter(self, optimizers=None):
        phase_real_img = self.input[0]
        phase_real_c = self.input[1]
        phases_all_gen_c = self.input[2]

        # 对齐梯度用
        # print('======================== batch%.5d.npz ========================'%self.batch_idx)
        # # dic2 = np.load('batch%.5d.npz'%self.batch_idx)
        # dic2 = np.load('tools/batch%.5d.npz'%self.batch_idx)
        # aaaaaaaaa = dic2['phase_real_img']
        # phase_real_img = torch.Tensor(aaaaaaaaa).cuda().to(torch.float32)

        phase_real_img = phase_real_img / 127.5 - 1


        phases = self.phases
        batch_size = phase_real_img.shape[0]

        all_gen_z = None
        num_gpus = 1  # 显卡数量
        batch_gpu = batch_size // num_gpus  # 一张显卡上的批大小
        if self.z_dim > 0:
            all_gen_z = torch.randn([len(phases) * batch_size, self.z_dim], device=phase_real_img.device)  # 咩酱：训练的4个阶段每个gpu的噪声
            # bbbbbbbbb = dic2['all_gen_z']
            # all_gen_z = torch.Tensor(bbbbbbbbb).cuda().to(torch.float32)
        else:
            all_gen_z = torch.randn([len(phases) * batch_size, 1], device=phase_real_img.device)  # 咩酱：训练的4个阶段每个gpu的噪声
        phases_all_gen_z = all_gen_z.split(batch_size)  # 咩酱：训练的4个阶段的噪声
        all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in phases_all_gen_z]  # 咩酱：训练的4个阶段每个gpu的噪声

        c_dim = phases_all_gen_c[0].shape[1]
        all_gen_c = None
        if c_dim > 0:
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in phases_all_gen_c]  # 咩酱：训练的4个阶段每个gpu的类别
        else:
            all_gen_c = [[None for _2 in range(num_gpus)] for _1 in range(len(phases))]

        phase_real_img = phase_real_img.split(batch_gpu)

        c_dim = phase_real_c.shape[1]
        if c_dim > 0:
            phase_real_c = phase_real_c.split(batch_gpu)
        else:
            phase_real_c = [[None for _2 in range(num_gpus)] for _1 in range(len(phases))]

        # Execute training phases.  咩酱：训练的4个阶段。一个批次的图片训练4个阶段。
        loss_numpys = dict()
        loss_phase_name = []
        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):  # 咩酱：phase_gen_z是这个阶段每个gpu的噪声，是一个元组，元组长度等于gpu数量。
            if self.batch_idx % phase['interval'] != 0:  # 咩酱：每一个阶段phase有一个属性interval，即训练间隔，每隔几个批次图片才会执行1次这个阶段！
                continue

            # Initialize gradient accumulation.  咩酱：初始化梯度累加（变相增大批大小）。
            # self._reset_grad(optimizers)  # 梯度清0
            # if 'G' in phase['name']:
            #     for name, param in self.nets['synthesis'].named_parameters():
            #         param.stop_gradient = False
            #     for name, param in self.nets['mapping'].named_parameters():
            #         param.stop_gradient = False
            #     for name, param in self.nets['discriminator'].named_parameters():
            #         param.stop_gradient = True
            # elif 'D' in phase['name']:
            #     for name, param in self.nets['synthesis'].named_parameters():
            #         param.stop_gradient = True
            #     for name, param in self.nets['mapping'].named_parameters():
            #         param.stop_gradient = True
            #     for name, param in self.nets['discriminator'].named_parameters():
            #         param.stop_gradient = False
            if 'G' in phase['name']:
                optimizers['optimizer_G'].zero_grad(set_to_none=True)
                self.mapping.requires_grad_(True)
                self.synthesis.requires_grad_(True)
            elif 'D' in phase['name']:
                optimizers['optimizer_D'].zero_grad(set_to_none=True)
                self.discriminator.requires_grad_(True)

            # 梯度累加。一个总的批次的图片分开{显卡数量}次遍历。
            # Accumulate gradients over multiple rounds.  咩酱：遍历每一个gpu上的批次图片。这样写好奇葩啊！round_idx是gpu_id
            for round_idx, (real_img, real_c, gen_z, gen_c) in enumerate(zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c)):
                sync = (round_idx == batch_size // (batch_gpu * num_gpus) - 1)   # 咩酱：右边的式子结果一定是0。即只有0号gpu做同步。这是梯度累加的固定写法。
                gain = phase['interval']     # 咩酱：即上文提到的训练间隔。

                # 梯度累加（变相增大批大小）。
                # loss_numpy = self.accumulate_gradients(phase=phase['name'], real_img=real_img, real_c=real_c,
                #                                        gen_z=gen_z, gen_c=gen_c, sync=sync, gain=gain, dic2=dic2)
                loss_numpy = self.accumulate_gradients(phase=phase['name'], real_img=real_img, real_c=real_c,
                                                       gen_z=gen_z, gen_c=gen_c, sync=sync, gain=gain)
                for k, v in loss_numpy.items():
                    if k in loss_numpys:
                        loss_numpys[k] += v
                    else:
                        loss_numpys[k] = v
                loss_phase_name.append(phase['name'])

            # Update weights.
            # phase.module.requires_grad_(False)
            # 梯度裁剪
            # for param in phase.module.parameters():
            #     if param.grad is not None:
            #         misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
            if 'G' in phase['name']:
                self.mapping.requires_grad_(False)
                self.synthesis.requires_grad_(False)
                optimizers['optimizer_G'].step()  # 更新参数
            elif 'D' in phase['name']:
                self.discriminator.requires_grad_(False)
                optimizers['optimizer_D'].step()  # 更新参数

        # compute moving average of network parameters。指数滑动平均
        # soft_update(self.synthesis,
        #             self.synthesis_ema,
        #             beta=0.999)
        # soft_update(self.mapping,
        #             self.mapping_ema,
        #             beta=0.999)
        self.batch_idx += 1
        return loss_numpys

    def test_iter(self, metrics=None):
        z = self.input['z']

        class_idx = None
        label = torch.zeros([1, self.c_dim], device=z.device)
        if self.c_dim != 0:
            if class_idx is None:
                print('Must specify class label with --class when using a conditional network')
            label[:, class_idx] = 1
        else:
            if class_idx is not None:
                print('warn: --class=lbl ignored when running on an unconditional network')
        # noise_mode = ['const', 'random', 'none']
        noise_mode = 'const'
        truncation_psi = 1.0

        ws = self.mapping_ema(z, label, truncation_psi=truncation_psi, truncation_cutoff=None)
        img = self.synthesis_ema(ws, noise_mode=noise_mode)

        img = img.permute((0, 2, 3, 1)) * 127.5 + 128
        img = img.clamp(0, 255)
        img = img.to(torch.uint8)
        img_rgb = img.cpu().detach().numpy()[0]
        img_bgr = img_rgb[:, :, [2, 1, 0]]
        return img_bgr
