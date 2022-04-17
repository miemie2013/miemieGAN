import cv2
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import contextlib

from mmgan.models.generators.generator_styleganv2ada import constant
from mmgan.utils import training_stats
from mmgan.utils.training_stats import EasyDict


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


def soft_update(source, target, beta=1.0):
    assert 0.0 <= beta <= 1.0
    for param, param_test in zip(source.parameters(), target.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


class StyleGANv2ADAModel:
    def __init__(
        self,
        synthesis,
        synthesis_ema,
        mapping,
        mapping_ema,
        discriminator=None,
        device=None,
        rank=0,
        G_reg_interval=4,
        D_reg_interval=16,
        augment_pipe=None,
        style_mixing_prob=0.9,
        r1_gamma=10,
        pl_batch_shrink=2,
        pl_decay=0.01,
        pl_weight=2.0,
        ema_kimg=10,
        ema_rampup=None,
        augment_p=0.0,
        ada_kimg=500,
        ada_interval=4,
        ada_target=None,
        adjust_p=False,
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
        self.synthesis.train()
        self.mapping.train()
        self.synthesis_ema.eval()
        self.mapping_ema.eval()
        if discriminator:
            self.discriminator = discriminator
            self.discriminator.train()
        self.device = device
        self.rank = rank
        self.c_dim = mapping.c_dim
        self.z_dim = mapping.z_dim
        self.w_dim = mapping.w_dim

        self.phases = []
        for name, reg_interval in [('G', G_reg_interval), ('D', D_reg_interval)]:
            if reg_interval is None:
                self.phases += [EasyDict(name=name + 'both', interval=1)]
            else:  # Lazy regularization.
                self.phases += [EasyDict(name=name + 'main', interval=1)]
                self.phases += [EasyDict(name=name + 'reg', interval=reg_interval)]
        for phase in self.phases:
            phase.start_event = None
            phase.end_event = None
            if rank == 0:
                phase.start_event = torch.cuda.Event(enable_timing=True)
                phase.end_event = torch.cuda.Event(enable_timing=True)

        self.z_dim = self.mapping.z_dim
        self.cur_nimg = 0
        self.batch_idx = 0

        # loss config.
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        # self.style_mixing_prob = -1.0
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight

        self.pl_mean = None
        self.ema_kimg = ema_kimg
        self.ema_rampup = ema_rampup

        self.augment_p = augment_p
        self.ada_kimg = ada_kimg
        self.ada_target = ada_target
        self.ada_interval = ada_interval
        self.adjust_p = adjust_p
        self.ada_stats = None
        if self.adjust_p:
            self.ada_stats = training_stats.Collector(regex='Loss/signs/real')

        self.align_grad = False
        # self.align_grad = True

        self.is_distributed = False



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

    def run_G(self, z, c, sync):
        '''
        除了self.augment_pipe，其它3个 self.G_mapping、self.G_synthesis、self.D 都是DDP模型。
        只有DDP模型才能使用with module.no_sync():
        '''
        with ddp_sync(self.mapping, sync):
            ws = self.mapping(z, c)
            if self.style_mixing_prob > 0:
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]

        with ddp_sync(self.synthesis, sync):
            img = self.synthesis(ws)
        return img, ws

    def run_D(self, img, c, sync):
        '''
        除了self.augment_pipe，其它3个 self.G_mapping、self.G_synthesis、self.D 都是DDP模型。
        只有DDP模型才能使用with module.no_sync():
        '''
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
            # debug_percentile = 0.7
            # img = self.augment_pipe(img, debug_percentile)
        with ddp_sync(self.discriminator, sync):
            logits = self.discriminator(img, c)
        return logits

    # 梯度累加（变相增大批大小）。dic2是为了梯度对齐。
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain, dic=None):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        loss_numpy = {}
        aaaaaaaa1 = training_stats._counters

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            # 训练生成器，判别器应该冻结，而且希望fake_img的gen_logits越大越好（愚弄D，使其判断是真图片），所以损失是-log(sigmoid(gen_logits))
            # 每个step都做1次
            gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl)) # May get synced by Gpl.
            # if self.align_grad:
            #     print_diff(dic, phase + ' gen_img', gen_img)
            #     print_diff(dic, phase + ' _gen_ws', _gen_ws)

            gen_logits = self.run_D(gen_img, gen_c, sync=False)
            # if self.align_grad:
            #     print_diff(dic, phase + ' gen_logits', gen_logits)

            training_stats.report('Loss/scores/fake', gen_logits)
            training_stats.report('Loss/signs/fake', gen_logits.sign())
            loss_Gmain = torch.nn.functional.softplus(-gen_logits)  # -log(sigmoid(gen_logits))
            training_stats.report('Loss/G/loss', loss_Gmain)
            # loss_Gmain = loss_Gmain.mean()
            # loss_numpy['loss_Gmain'] = loss_Gmain.cpu().detach().numpy()

            # loss_G = loss_Gmain
            # loss_G = loss_G * float(gain)
            # loss_G.backward()  # 咩酱：gain即上文提到的这个阶段的训练间隔。
            loss_Gmain.mean().mul(gain).backward()
            # if self.align_grad:
            #     mapping = self.mapping.module if self.is_distributed else self.mapping
            #     synthesis = self.synthesis.module if self.is_distributed else self.synthesis
            #     discriminator = self.discriminator.module if self.is_distributed else self.discriminator
            #     m_w_grad = mapping.fc7.weight.grad
            #     m_b_grad = mapping.fc7.bias.grad
            #     s_w_grad = synthesis.b32.conv0.affine.weight.grad
            #     s_b_grad = synthesis.b32.conv0.affine.bias.grad
            #     d_w_grad = discriminator.b32.conv0.weight.grad
            #     d_b_grad = discriminator.b32.conv0.bias.grad
            #     print_diff(dic, phase + ' m_w_grad', m_w_grad)
            #     print_diff(dic, phase + ' m_b_grad', m_b_grad)
            #     print_diff(dic, phase + ' s_w_grad', s_w_grad)
            #     print_diff(dic, phase + ' s_b_grad', s_b_grad)
            #     print_diff(dic, phase + ' d_w_grad', d_w_grad)
            #     print_diff(dic, phase + ' d_b_grad', d_b_grad)

        # Gpl: Apply path length regularization.
        if do_Gpl:
            # 训练生成器，判别器应该冻结（其实也没有跑判别器），是生成器的梯度惩罚损失（一种高级一点的梯度裁剪）
            # 每4个step做1次
            batch_size = gen_z.shape[0] // self.pl_batch_shrink
            batch_size = max(batch_size, 1)

            gen_c_ = None
            if gen_c is not None:
                gen_c_ = gen_c[:batch_size]

            gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c_, sync=sync)
            # if self.align_grad:
            #     print_diff(dic, phase + ' gen_img', gen_img)
            #     print_diff(dic, phase + ' gen_ws', gen_ws)
            pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
            # pl_noise = torch.ones_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
            with no_weight_gradients():
                pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]

            pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
            # if self.align_grad:
            #     print_diff(dic, phase + ' pl_grads', pl_grads)
            if self.pl_mean is None:
                self.pl_mean = torch.zeros([1, ], dtype=torch.float32, device=self.device)
            pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
            self.pl_mean.copy_(pl_mean.detach())

            pl_penalty = (pl_lengths - pl_mean).square()
            training_stats.report('Loss/pl_penalty', pl_penalty)
            loss_Gpl = pl_penalty * self.pl_weight
            training_stats.report('Loss/G/reg', loss_Gpl)

            # loss_Gpl = (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean() * float(gain)
            # loss_numpy['loss_Gpl'] = loss_Gpl.cpu().detach().numpy()
            # loss_Gpl.backward()  # 咩酱：gain即上文提到的这个阶段的训练间隔。
            (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()
            # if self.align_grad:
            #     mapping = self.mapping.module if self.is_distributed else self.mapping
            #     synthesis = self.synthesis.module if self.is_distributed else self.synthesis
            #     discriminator = self.discriminator.module if self.is_distributed else self.discriminator
            #     m_w_grad = mapping.fc7.weight.grad
            #     m_b_grad = mapping.fc7.bias.grad
            #     s_w_grad = synthesis.b32.conv0.affine.weight.grad
            #     s_b_grad = synthesis.b32.conv0.affine.bias.grad
            #     d_w_grad = discriminator.b32.conv0.weight.grad
            #     d_b_grad = discriminator.b32.conv0.bias.grad
            #     print_diff(dic, phase + ' m_w_grad', m_w_grad)
            #     print_diff(dic, phase + ' m_b_grad', m_b_grad)
            #     print_diff(dic, phase + ' s_w_grad', s_w_grad)
            #     print_diff(dic, phase + ' s_b_grad', s_b_grad)
            #     print_diff(dic, phase + ' d_w_grad', d_w_grad)
            #     print_diff(dic, phase + ' d_b_grad', d_b_grad)

        # Dmain: Minimize logits for generated images.
        if do_Dmain:
            # 训练判别器，生成器应该冻结，而且希望fake_img的gen_logits越小越好（判断是假图片），所以损失是-log(1 - sigmoid(gen_logits))
            # 每个step都做1次
            gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=False)
            # if self.align_grad:
            #     print_diff(dic, phase + ' gen_img', gen_img)
            #     print_diff(dic, phase + ' _gen_ws', _gen_ws)
            gen_logits = self.run_D(gen_img, gen_c, sync=False) # Gets synced by loss_Dreal.
            # if self.align_grad:
            #     print_diff(dic, phase + ' gen_logits', gen_logits)
            training_stats.report('Loss/scores/fake', gen_logits)
            training_stats.report('Loss/signs/fake', gen_logits.sign())

            loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            # loss_Dgen = loss_Dgen.mean()
            # loss_numpy['loss_Dgen'] = loss_Dgen.cpu().detach().numpy()

            # loss3 = loss_Dgen * float(gain)
            # loss3.backward()  # 咩酱：gain即上文提到的这个阶段的训练间隔。
            loss_Dgen.mean().mul(gain).backward()
            # if self.align_grad:
            #     mapping = self.mapping.module if self.is_distributed else self.mapping
            #     synthesis = self.synthesis.module if self.is_distributed else self.synthesis
            #     discriminator = self.discriminator.module if self.is_distributed else self.discriminator
            #     m_w_grad = mapping.fc7.weight.grad
            #     m_b_grad = mapping.fc7.bias.grad
            #     s_w_grad = synthesis.b32.conv0.affine.weight.grad
            #     s_b_grad = synthesis.b32.conv0.affine.bias.grad
            #     d_w_grad = discriminator.b32.conv0.weight.grad
            #     d_b_grad = discriminator.b32.conv0.bias.grad
            #     print_diff(dic, phase + ' backward0 m_w_grad', m_w_grad)
            #     print_diff(dic, phase + ' backward0 m_b_grad', m_b_grad)
            #     print_diff(dic, phase + ' backward0 s_w_grad', s_w_grad)
            #     print_diff(dic, phase + ' backward0 s_b_grad', s_b_grad)
            #     print_diff(dic, phase + ' backward0 d_w_grad', d_w_grad)
            #     print_diff(dic, phase + ' backward0 d_b_grad', d_b_grad)

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
            real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
            training_stats.report('Loss/scores/real', real_logits)
            training_stats.report('Loss/signs/real', real_logits.sign())
            # if self.align_grad:
            #     print_diff(dic, phase + ' real_logits', real_logits)

            loss_Dreal = 0
            if do_Dmain:
                # 训练判别器，生成器应该冻结，而且希望real_img的gen_logits越大越好（判断是真图片），所以损失是-log(sigmoid(real_logits))
                # 每个step都做1次
                loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                # if self.align_grad:
                #     print_diff(dic, phase + ' loss_Dreal', loss_Dreal)
                loss_numpy['loss_Dreal'] = loss_Dreal.cpu().detach().numpy().mean()
                training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

            loss_Dr1 = 0
            if do_Dr1:
                # 训练判别器，生成器应该冻结（其实也没有跑判别器），是判别器的梯度惩罚损失（一种高级一点的梯度裁剪）
                # 每16个step做1次
                with no_weight_gradients():
                    r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                # if self.align_grad:
                #     print_diff(dic, phase + ' r1_grads', r1_grads)
                r1_penalty = r1_grads.square().sum([1, 2, 3])
                loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                loss_numpy['loss_Dr1'] = loss_Dr1.cpu().detach().numpy().mean()
                training_stats.report('Loss/r1_penalty', r1_penalty)
                training_stats.report('Loss/D/reg', loss_Dr1)

            # loss4 = (loss_Dreal + loss_Dr1).mean() * float(gain)
            # if do_Dmain:
            #     loss4 += loss3
            # loss4.backward()  # 咩酱：gain即上文提到的这个阶段的训练间隔。
            (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()
            # if self.align_grad:
            #     mapping = self.mapping.module if self.is_distributed else self.mapping
            #     synthesis = self.synthesis.module if self.is_distributed else self.synthesis
            #     discriminator = self.discriminator.module if self.is_distributed else self.discriminator
            #     m_w_grad = mapping.fc7.weight.grad
            #     m_b_grad = mapping.fc7.bias.grad
            #     s_w_grad = synthesis.b32.conv0.affine.weight.grad
            #     s_b_grad = synthesis.b32.conv0.affine.bias.grad
            #     d_w_grad = discriminator.b32.conv0.weight.grad
            #     d_b_grad = discriminator.b32.conv0.bias.grad
            #     if do_Dmain:
            #         print_diff(dic, phase + ' backward1 m_w_grad', m_w_grad)
            #         print_diff(dic, phase + ' backward1 m_b_grad', m_b_grad)
            #         print_diff(dic, phase + ' backward1 s_w_grad', s_w_grad)
            #         print_diff(dic, phase + ' backward1 s_b_grad', s_b_grad)
            #         print_diff(dic, phase + ' backward1 d_w_grad', d_w_grad)
            #         print_diff(dic, phase + ' backward1 d_b_grad', d_b_grad)
            #     if do_Dr1:
            #         print_diff(dic, phase + ' m_w_grad', m_w_grad)
            #         print_diff(dic, phase + ' m_b_grad', m_b_grad)
            #         print_diff(dic, phase + ' s_w_grad', s_w_grad)
            #         print_diff(dic, phase + ' s_b_grad', s_b_grad)
            #         print_diff(dic, phase + ' d_w_grad', d_w_grad)
            #         print_diff(dic, phase + ' d_b_grad', d_b_grad)
        return loss_numpy

    def train_iter(self, optimizers=None, rank=0, world_size=1):
        device = self.device
        phase_real_img = self.input[0]
        phase_real_c = self.input[1]
        phases_all_gen_c = self.input[2]

        # if self.align_grad:
        #     if self.batch_idx == 0:
        #         '''
        #         假如训练命令的命令行参数是 --gpus=2 --batch 8 --cfg my32
        #         即总的批大小是8，每卡批大小是4，那么这里
        #         phase_real_img.shape = [4, 3, 32, 32]
        #         phase_real_c.shape   = [4, 0]
        #         batch_gpu            = 4
        #         即拿到的phase_real_img和phase_real_c是一张卡（一个进程）上的训练样本，（每张卡）批大小是4
        #         '''
        #         print('rank =', rank)
        #         print('world_size =', world_size)
        #         print('phase_real_img.shape =', phase_real_img.shape)
        #         print('phase_real_c.shape =', phase_real_c.shape)

        # 对齐梯度用
        dic2 = None
        # if self.align_grad:
        #     print('======================== batch%.5d.npz ========================'%self.batch_idx)
        #     npz_path = 'batch%.5d_rank%.2d.npz'%(self.batch_idx, rank)
        #     isDebug = True if sys.gettrace() else False
        #     if isDebug:
        #         npz_path = '../batch%.5d_rank%.2d.npz'%(self.batch_idx, rank)
        #     dic2 = np.load(npz_path)
        #     aaaaaaaaa = dic2['phase_real_img']
        #     phase_real_img = torch.Tensor(aaaaaaaaa).to(device).to(torch.float32)

        phase_real_img = phase_real_img.to(device).to(torch.float32) / 127.5 - 1


        phases = self.phases
        batch_gpu = phase_real_img.shape[0]  # 一张显卡上的批大小

        all_gen_z = None
        num_gpus = world_size  # 显卡数量
        batch_size = batch_gpu * num_gpus
        if self.z_dim > 0:
            all_gen_z = torch.randn([len(phases) * batch_size, self.z_dim], device=phase_real_img.device)  # 咩酱：训练的4个阶段每个gpu的噪声
            # if self.align_grad:
            #     all_gen_z = torch.Tensor(dic2['all_gen_z']).to(device).to(torch.float32)
        else:
            all_gen_z = torch.randn([len(phases) * batch_size, 1], device=phase_real_img.device)  # 咩酱：训练的4个阶段每个gpu的噪声
        phases_all_gen_z = all_gen_z.split(batch_size)  # 咩酱：训练的4个阶段的噪声
        all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in phases_all_gen_z]  # 咩酱：训练的4个阶段每个gpu的噪声

        c_dim = phases_all_gen_c[0].shape[1]
        all_gen_c = None
        if c_dim > 0:
            all_gen_c = [phase_gen_c.to(device).split(batch_gpu) for phase_gen_c in phases_all_gen_c]  # 咩酱：训练的4个阶段每个gpu的类别
        else:
            all_gen_c = [[None for _2 in range(num_gpus)] for _1 in range(len(phases))]

        phase_real_img = phase_real_img.split(batch_gpu)

        c_dim = phase_real_c.shape[1]
        if c_dim > 0:
            phase_real_c = phase_real_c.to(device).split(batch_gpu)
        else:
            phase_real_c = [[None for _2 in range(num_gpus)] for _1 in range(len(phases))]

        # Execute training phases.  咩酱：训练的4个阶段。一个批次的图片训练4个阶段。
        loss_numpys = dict()
        loss_phase_name = []
        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):  # 咩酱：phase_gen_z是这个阶段每个gpu的噪声，是一个元组，元组长度等于gpu数量。
            if self.batch_idx % phase.interval != 0:  # 咩酱：每一个阶段phase有一个属性interval，即训练间隔，每隔几个批次图片才会执行1次这个阶段！
                continue

            # Initialize gradient accumulation.  咩酱：初始化梯度累加（变相增大批大小）。
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))
            if 'G' in phase.name:
                optimizers['optimizer_G'].zero_grad(set_to_none=True)
                self.mapping.requires_grad_(True)
                self.synthesis.requires_grad_(True)
                # for param_group in optimizers['optimizer_G'].param_groups:
                #     param_group["params"][0].requires_grad = True
            elif 'D' in phase.name:
                optimizers['optimizer_D'].zero_grad(set_to_none=True)
                self.discriminator.requires_grad_(True)
                # for param_group in optimizers['optimizer_D'].param_groups:
                #     param_group["params"][0].requires_grad = True

            # 梯度累加。不管多卡还是单卡，这个for循环只会循环1次。
            # Accumulate gradients over multiple rounds.
            for round_idx, (real_img, real_c, gen_z, gen_c) in enumerate(zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c)):
                sync = (round_idx == batch_size // (batch_gpu * num_gpus) - 1)   # 咩酱：右边的式子结果一定是0。sync一定是True
                gain = phase.interval     # 咩酱：即上文提到的训练间隔。

                # 梯度累加（变相增大批大小）。
                loss_numpy = self.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c,
                                                       gen_z=gen_z, gen_c=gen_c, sync=sync, gain=gain, dic=dic2)
                for k, v in loss_numpy.items():
                    if k in loss_numpys:
                        loss_numpys[k] += v
                    else:
                        loss_numpys[k] = v
                loss_phase_name.append(phase.name)

            # Update weights.
            # phase.module.requires_grad_(False)
            # 梯度裁剪
            # for param in phase.module.parameters():
            #     if param.grad is not None:
            #         misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
            if 'G' in phase.name:
                self.mapping.requires_grad_(False)
                self.synthesis.requires_grad_(False)
                # for param_group in optimizers['optimizer_G'].param_groups:
                #     param_group["params"][0].requires_grad = False
                optimizers['optimizer_G'].step()  # 更新参数
            elif 'D' in phase.name:
                self.discriminator.requires_grad_(False)
                # for param_group in optimizers['optimizer_D'].param_groups:
                #     param_group["params"][0].requires_grad = False
                optimizers['optimizer_D'].step()  # 更新参数
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # compute moving average of network parameters。指数滑动平均
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

        # if self.align_grad:
        #     if self.is_distributed:
        #         w_avg = self.mapping.module.w_avg
        #     else:
        #         w_avg = self.mapping.w_avg
        #     print_diff(dic2, 'w_avg', w_avg)

        # Execute ADA heuristic.
        if self.adjust_p and self.augment_pipe is not None and (self.batch_idx % self.ada_interval == 0):
            # self.ada_interval个迭代中，real_logits.sign()的平均值。
            self.ada_stats.update()
            Loss_signs_real_mean = self.ada_stats['Loss/signs/real']

            diff = Loss_signs_real_mean - self.ada_target
            adjust = np.sign(diff)
            # if self.align_grad:
            #     print_diff(dic2, 'augment_pipe_p', self.augment_pipe.p)
            #     kkk = 'aaaaaaaaaa1'; ddd = np.sum((dic2[kkk] - np.array(Loss_signs_real_mean)) ** 2)
            #     print('diff=%.6f (%s)' % (ddd, kkk))
            adjust = adjust * (batch_size * self.ada_interval) / (self.ada_kimg * 1000)
            self.augment_pipe.p.copy_((self.augment_pipe.p + adjust).max(constant(0, device=device)))

        return loss_numpys

    def test_iter(self, metrics=None):
        z = self.input['z']
        seed = self.input['seed']
        seed = seed.cpu().detach().numpy()[0]

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
        return img_bgr, seed

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
