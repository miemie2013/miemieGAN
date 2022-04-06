import cv2
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

from mmgan.models.generators.generator_styleganv2ada import constant
from mmgan.models.generators.generator_styleganv3 import filter2d


def soft_update(source, target, beta=1.0):
    assert 0.0 <= beta <= 1.0
    for param, param_test in zip(source.parameters(), target.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


class StyleGANv3Model(torch.nn.Module):
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
        style_mixing_prob=0.0,
        r1_gamma=10,
        pl_batch_shrink=2,
        pl_decay=0.01,
        pl_weight=0.0,
        ema_kimg=10,
        ema_rampup=None,
        augment_p=0.0,
        ada_kimg=500,
        ada_interval=4,
        ada_target=None,
        pl_no_weight_grad=False,
        blur_init_sigma=0,
        blur_fade_kimg=0,
        adjust_p=False,
    ):
        super(StyleGANv3Model, self).__init__()
        self.optimizers = OrderedDict()
        self.metrics = OrderedDict()
        self.losses = OrderedDict()
        self.visual_items = OrderedDict()

        self.synthesis = synthesis
        self.synthesis_ema = synthesis_ema
        self.mapping = mapping
        self.mapping_ema = mapping_ema
        self.synthesis.train()
        self.mapping.train()
        self.synthesis_ema.eval()
        self.mapping_ema.eval()
        if discriminator:
            self.discriminator = discriminator
            self.discriminator.train()
        self.c_dim = mapping.c_dim
        self.z_dim = mapping.z_dim
        self.w_dim = mapping.w_dim

        self.phases = []
        for name, reg_interval in [('G', G_reg_interval), ('D', D_reg_interval)]:
            if reg_interval is None:
                self.phases += [dict(name=name + 'both', interval=1)]
            else:  # Lazy regularization.
                self.phases += [dict(name=name + 'main', interval=1)]
                self.phases += [dict(name=name + 'reg', interval=reg_interval)]

        self.z_dim = self.mapping.z_dim
        self.cur_nimg = 0
        self.batch_idx = 0

        # loss config.
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        # self.augment_pipe = None
        # self.style_mixing_prob = -1.0
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_no_weight_grad = pl_no_weight_grad
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg

        self.pl_mean = None
        self.ema_kimg = ema_kimg
        self.ema_rampup = ema_rampup

        self.augment_p = augment_p
        self.ada_kimg = ada_kimg
        self.ada_target = ada_target
        self.ada_interval = ada_interval
        self.adjust_p = adjust_p
        self.Loss_signs_real = []

        self.align_grad = False
        # self.align_grad = True



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

    def run_G(self, z, c, update_emas=False):
        ws = self.mapping(z, c, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
            cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
            ws[:, cutoff:] = self.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]

        img = self.synthesis(ws, update_emas=update_emas)
        return img, ws

    def run_D(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
            img = filter2d(img, f / f.sum())
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        logits = self.discriminator(img, c, update_emas=update_emas)
        return logits

    # 梯度累加（变相增大批大小）。dic2是为了梯度对齐。
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg, dic2=None):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.pl_weight == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0

        loss_numpy = {}

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            gen_img, _gen_ws = self.run_G(gen_z, gen_c)
            # if self.align_grad:
            #     ddd = np.sum((dic2[phase + 'gen_img'] - gen_img.cpu().detach().numpy()) ** 2)
            #     print('do_Gmain gen_img=%.6f' % ddd)
            #     ddd = np.sum((dic2[phase + '_gen_ws'] - _gen_ws.cpu().detach().numpy()) ** 2)
            #     print('do_Gmain _gen_ws=%.6f' % ddd)

            gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
            # if self.align_grad:
            #     ddd = np.sum((dic2[phase + 'gen_logits'] - gen_logits.cpu().detach().numpy()) ** 2)
            #     print('do_Gmain gen_logits=%.6f' % ddd)

            loss_Gmain = torch.nn.functional.softplus(-gen_logits)  # -log(sigmoid(gen_logits))
            loss_Gmain = loss_Gmain.mean()
            loss_numpy['loss_Gmain'] = loss_Gmain.cpu().detach().numpy()

            loss_G = loss_Gmain
            loss_G = loss_G * float(gain)
            loss_G.backward()  # 咩酱：gain即上文提到的这个阶段的训练间隔。

        # Gpl: Apply path length regularization.
        if phase in ['Greg', 'Gboth']:
            batch_size = gen_z.shape[0] // self.pl_batch_shrink
            batch_size = max(batch_size, 1)

            gen_c_ = None
            if gen_c is not None:
                gen_c_ = gen_c[:batch_size]

            gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c_)
            # if self.align_grad:
            #     ddd = np.sum((dic2[phase + 'gen_img'] - gen_img.cpu().detach().numpy()) ** 2)
            #     print('do_Gpl gen_img=%.6f' % ddd)
            #     ddd = np.sum((dic2[phase + 'gen_ws'] - gen_ws.cpu().detach().numpy()) ** 2)
            #     print('do_Gpl gen_ws=%.6f' % ddd)
            pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
            # pl_noise = torch.ones_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
            pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]

            pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
            # if self.align_grad:
            #     ddd = np.sum((dic2[phase + 'pl_grads'] - pl_grads.cpu().detach().numpy()) ** 2)
            #     print('do_Gpl pl_grads=%.6f' % ddd)
            #     ddd = np.sum((dic2[phase + 'pl_lengths'] - pl_lengths.cpu().detach().numpy()) ** 2)
            #     print('do_Gpl pl_lengths=%.6f' % ddd)
            if self.pl_mean is None:
                self.pl_mean = torch.zeros([1, ], dtype=torch.float32, device=pl_lengths.device)
            pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
            self.pl_mean.copy_(pl_mean.detach())

            pl_penalty = (pl_lengths - pl_mean).square()
            loss_Gpl = pl_penalty * self.pl_weight

            loss_Gpl = loss_Gpl.mean() * float(gain)
            loss_numpy['loss_Gpl'] = loss_Gpl.cpu().detach().numpy()
            loss_Gpl.backward()  # 咩酱：gain即上文提到的这个阶段的训练间隔。

        # Dmain: Minimize logits for generated images.
        loss3 = 0.0
        if phase in ['Dmain', 'Dboth']:
            gen_img, _gen_ws = self.run_G(gen_z, gen_c, update_emas=True)
            # if self.align_grad:
            #     ddd = np.sum((dic2[phase + 'gen_img'] - gen_img.cpu().detach().numpy()) ** 2)
            #     print('do_Dmain gen_img=%.6f' % ddd)
            #     ddd = np.sum((dic2[phase + '_gen_ws'] - _gen_ws.cpu().detach().numpy()) ** 2)
            #     print('do_Dmain _gen_ws=%.6f' % ddd)
            gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True)
            # if self.align_grad:
            #     ddd = np.sum((dic2[phase + 'gen_logits'] - gen_logits.cpu().detach().numpy()) ** 2)
            #     print('do_Dmain gen_logits=%.6f' % ddd)

            loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            # loss_Dgen.mean().mul(gain).backward()
            loss_Dgen = loss_Dgen.mean()
            loss_numpy['loss_Dgen'] = loss_Dgen.cpu().detach().numpy()

            loss3 = loss_Dgen * float(gain)
            loss3.backward()  # 咩酱：gain即上文提到的这个阶段的训练间隔。

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            real_img_tmp = real_img.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
            real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
            if self.adjust_p and self.augment_pipe is not None:
                self.Loss_signs_real.append(real_logits.sign().cpu().detach().numpy())
            # if self.align_grad:
            #     ddd = np.sum((dic2[phase + 'real_logits'] - real_logits.cpu().detach().numpy()) ** 2)
            #     print('do_Dmain or do_Dr1 real_logits=%.6f' % ddd)

            loss_Dreal = 0
            if phase in ['Dmain', 'Dboth']:
                loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                # if self.align_grad:
                #     ddd = np.sum((dic2[phase + 'loss_Dreal'] - loss_Dreal.cpu().detach().numpy()) ** 2)
                #     print('do_Dmain or do_Dr1 do_Dmain loss_Dreal=%.6f' % ddd)
                loss_numpy['loss_Dreal'] = loss_Dreal.cpu().detach().numpy().mean()

            loss_Dr1 = 0
            if phase in ['Dreg', 'Dboth']:
                r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                # if self.align_grad:
                #     ddd = np.sum((dic2[phase + 'r1_grads'] - r1_grads.cpu().detach().numpy()) ** 2)
                #     print('do_Dmain or do_Dr1 do_Dr1 r1_grads=%.6f' % ddd)
                r1_penalty = r1_grads.square().sum([1, 2, 3])
                # if self.align_grad:
                #     ddd = np.sum((dic2[phase + 'r1_penalty'] - r1_penalty.cpu().detach().numpy()) ** 2)
                #     print('do_Dmain or do_Dr1 do_Dr1 r1_penalty=%.6f' % ddd)
                loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                loss_numpy['loss_Dr1'] = loss_Dr1.cpu().detach().numpy().mean()

            loss4 = (loss_Dreal + loss_Dr1).mean() * float(gain)
            # if do_Dmain:
            #     loss4 += loss3
            loss4.backward()  # 咩酱：gain即上文提到的这个阶段的训练间隔。
        return loss_numpy

    def train_iter(self, optimizers=None):
        phase_real_img = self.input[0]
        phase_real_c = self.input[1]
        phases_all_gen_c = self.input[2]

        # 对齐梯度用
        dic2 = None
        # if self.align_grad:
        #     print('======================== batch%.5d.npz ========================'%self.batch_idx)
        #     npz_path = 'batch%.5d.npz'%self.batch_idx
        #     isDebug = True if sys.gettrace() else False
        #     if isDebug:
        #         npz_path = '../batch%.5d.npz'%self.batch_idx
        #     dic2 = np.load(npz_path)
        #     aaaaaaaaa = dic2['phase_real_img']
        #     phase_real_img = torch.Tensor(aaaaaaaaa).cuda().to(torch.float32)

        phase_real_img = phase_real_img / 127.5 - 1


        phases = self.phases
        batch_size = phase_real_img.shape[0]

        all_gen_z = None
        num_gpus = 1  # 显卡数量
        batch_gpu = batch_size // num_gpus  # 一张显卡上的批大小
        if self.z_dim > 0:
            all_gen_z = torch.randn([len(phases) * batch_size, self.z_dim], device=phase_real_img.device)  # 咩酱：训练的4个阶段每个gpu的噪声
            # if self.align_grad:
            #     all_gen_z = torch.Tensor(dic2['all_gen_z']).cuda().to(torch.float32)
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
            if 'G' in phase['name']:
                optimizers['optimizer_G'].zero_grad(set_to_none=True)
                # for param_group in optimizers['optimizer_G'].param_groups:
                #     param_group["params"][0].requires_grad = True
                self.mapping.requires_grad_(True)
                self.synthesis.requires_grad_(True)
            elif 'D' in phase['name']:
                optimizers['optimizer_D'].zero_grad(set_to_none=True)
                # for param_group in optimizers['optimizer_D'].param_groups:
                #     param_group["params"][0].requires_grad = True
                self.discriminator.requires_grad_(True)

            # 梯度累加。一个总的批次的图片分开{显卡数量}次遍历。
            # Accumulate gradients over multiple rounds.  咩酱：遍历每一个gpu上的批次图片。这样写好奇葩啊！round_idx是gpu_id
            for round_idx, (real_img, real_c, gen_z, gen_c) in enumerate(zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c)):
                gain = phase['interval']     # 咩酱：即上文提到的训练间隔。

                # 梯度累加（变相增大批大小）。
                loss_numpy = self.accumulate_gradients(phase=phase['name'], real_img=real_img, real_c=real_c,
                                                       gen_z=gen_z, gen_c=gen_c, gain=gain, cur_nimg=self.cur_nimg, dic2=dic2)
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
                # for param_group in optimizers['optimizer_G'].param_groups:
                #     param_group["params"][0].requires_grad = False
                self.mapping.requires_grad_(False)
                self.synthesis.requires_grad_(False)
                optimizers['optimizer_G'].step()  # 更新参数
            elif 'D' in phase['name']:
                # for param_group in optimizers['optimizer_D'].param_groups:
                #     param_group["params"][0].requires_grad = False
                self.discriminator.requires_grad_(False)
                optimizers['optimizer_D'].step()  # 更新参数

        # compute moving average of network parameters。指数滑动平均
        self.mapping_ema.requires_grad_(False)
        self.synthesis_ema.requires_grad_(False)
        ema_kimg = self.ema_kimg
        ema_nimg = ema_kimg * 1000
        ema_rampup = self.ema_rampup
        cur_nimg = self.cur_nimg
        if ema_rampup is not None:
            ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
        ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
        for p_ema, p in zip(self.mapping_ema.parameters(), self.mapping.parameters()):
            p_ema.copy_(p.lerp(p_ema, ema_beta))   # p_ema = ema_beta * p_ema + (1 - ema_beta) * p   ;ema模型占的比重ema_beta大
        for b_ema, b in zip(self.mapping_ema.buffers(), self.mapping.buffers()):
            b_ema.copy_(b)
        for p_ema, p in zip(self.synthesis_ema.parameters(), self.synthesis.parameters()):
            p_ema.copy_(p.lerp(p_ema, ema_beta))   # p_ema = ema_beta * p_ema + (1 - ema_beta) * p   ;ema模型占的比重ema_beta大
        for b_ema, b in zip(self.synthesis_ema.buffers(), self.synthesis.buffers()):
            b_ema.copy_(b)

        self.cur_nimg += batch_size
        self.batch_idx += 1

        # Execute ADA heuristic.
        if self.adjust_p and self.augment_pipe is not None and (self.batch_idx % self.ada_interval == 0):
            # self.ada_interval个迭代中，real_logits.sign()的平均值。
            Loss_signs_real_mean = np.mean(np.concatenate(self.Loss_signs_real, 0))
            diff = Loss_signs_real_mean - self.ada_target
            adjust = np.sign(diff)
            # print(Loss_signs_real_mean)
            # print('==========================')
            adjust = adjust * (batch_size * self.ada_interval) / (self.ada_kimg * 1000)
            self.augment_pipe.p.copy_((self.augment_pipe.p + adjust).max(constant(0, device=self.augment_pipe.p.device)))
            self.Loss_signs_real = []

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

    def style_mixing(self, row_seeds, col_seeds, all_seeds, col_styles):
        all_z = self.input['z']
        # noise_mode = ['const', 'random', 'none']
        noise_mode = 'const'
        truncation_psi = 1.0
        all_w = self.mapping_ema(all_z, None)
        w_avg = self.mapping_ema.w_avg
        all_w = w_avg + (all_w - w_avg) * truncation_psi
        w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))}

        # print('Generating images...')
        all_images = self.synthesis_ema(all_w, noise_mode=noise_mode)
        all_images = (all_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
        image_dict = {(seed, seed): image for seed, image in zip(all_seeds, list(all_images))}

        # print('Generating style-mixed images...')
        for row_seed in row_seeds:
            for col_seed in col_seeds:
                w = w_dict[row_seed].clone()
                w[col_styles] = w_dict[col_seed][col_styles]
                image = self.synthesis_ema(w[np.newaxis], noise_mode=noise_mode)
                image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                image_dict[(row_seed, col_seed)] = image[0].cpu().numpy()

        # print('Saving image grid...')
        ROW = len(row_seeds)
        COL = len(col_seeds)
        res = self.synthesis_ema.img_resolution
        grid_img_rgb = np.zeros(((ROW+1)*res, (COL+1)*res, 3), dtype=np.uint8)
        for j, row_seed in enumerate(row_seeds):
            for i, col_seed in enumerate(col_seeds):
                grid_img_rgb[(j+1)*res:(j+2)*res, (i+1)*res:(i+2)*res, :] = image_dict[(row_seed, col_seed)]
        for j, row_seed in enumerate(row_seeds):
            grid_img_rgb[(j+1)*res:(j+2)*res, 0:res, :] = image_dict[(row_seed, row_seed)]
        for i, col_seed in enumerate(col_seeds):
            grid_img_rgb[0:res, (i+1)*res:(i+2)*res, :] = image_dict[(col_seed, col_seed)]
        grid_img_bgr = grid_img_rgb[:, :, [2, 1, 0]]
        return grid_img_bgr
