#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @miemie2013

import os
import sys
import random

import torch
import torch.distributed as dist
import torch.nn as nn

from mmgan.data import *
from mmgan.exp.base_exp import BaseExp



class StyleGANv3_Method_Exp(BaseExp):
    def __init__(self):
        super().__init__()
        # ---------------- architecture name(算法名) ---------------- #
        self.archi_name = 'StyleGANv3'

        # --------------  training config --------------------- #
        self.G_reg_interval = None
        self.D_reg_interval = 16

        self.max_epoch = None
        self.kimgs = 25000
        self.print_interval = 10
        self.temp_img_interval = 100
        self.eval_interval = 1
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # learning_rate
        self.basic_glr_per_img = 0.0025 / 32.0
        self.basic_dlr_per_img = 0.002 / 32.0
        self.optimizer_cfg = dict(
            generator=dict(
                beta1=0.0,
                beta2=0.99,
                epsilon=1e-8,
            ),
            discriminator=dict(
                beta1=0.0,
                beta2=0.99,
                epsilon=1e-8,
            ),
        )

        # -----------------  testing config ------------------ #
        self.noise_mode = 'const'   # ['const', 'random', 'none']
        self.truncation_psi = 1.0

        # ---------------- model config ---------------- #
        self.output_dir = "StyleGANv3_outputs"
        self.w_dim = 512
        self.z_dim = 512
        self.c_dim = 0
        self.img_resolution = 512
        self.img_channels = 3
        self.channel_base = 32768
        self.channel_max = 512
        self.synthesis_freeze_at = []
        self.discriminator_freeze_at = []
        self.stylegan_cfg = 'stylegan3-r'
        self.synthesis_type = 'StyleGANv3_SynthesisNetwork'
        self.synthesis = dict(
            w_dim=self.w_dim,
            img_resolution=self.img_resolution,
            img_channels=self.img_channels,
            channel_base=self.channel_base,
            channel_max=self.channel_max,
            magnitude_ema_beta=0.999,
        )
        self.mapping_type = 'StyleGANv3_MappingNetwork'
        self.mapping = dict(
            z_dim=self.z_dim,
            c_dim=self.c_dim,
            w_dim=self.w_dim,
            num_layers=2,
        )
        self.discriminator_type = 'StyleGANv3_Discriminator'
        self.discriminator = dict(
            c_dim=self.c_dim,
            img_resolution=self.img_resolution,
            img_channels=self.img_channels,
            channel_base=self.channel_base,
            channel_max=self.channel_max,
            block_kwargs=dict(freeze_layers=0,),
            mapping_kwargs={},
            epilogue_kwargs=dict(mbstd_group_size=None,),   # default is 4
        )
        self.augment_pipe_type = 'StyleGANv2ADA_AugmentPipe'
        self.augment_pipe = dict(
            xflip=1,
            rotate90=1,
            xint=1,
            scale=1,
            rotate=1,
            aniso=1,
            xfrac=1,
            brightness=1,
            contrast=1,
            lumaflip=1,
            hue=1,
            saturation=1,
        )
        self.model_cfg = dict(
            G_reg_interval=self.G_reg_interval,
            D_reg_interval=self.D_reg_interval,
            r1_gamma=16.4,
            pl_batch_shrink=2,
            ema_kimg=-1,
            ema_rampup=None,
            augment_p=0.0,
            ada_kimg=100,
            ada_interval=4,
            ada_target=0.6,
        )

        # ---------------- dataset config ---------------- #
        self.dataroot = '../data/data42681/afhq/train/cat'
        # self.dataroot = '../data/data42681/afhq/train/dog'
        len_phases = 2
        if self.G_reg_interval is not None:
            len_phases += 1
        if self.D_reg_interval is not None:
            len_phases += 1
        self.dataset_train_cfg = dict(
            resolution=self.img_resolution,
            use_labels=False,
            xflip=True,
            len_phases=len_phases,
        )
        self.dataset_test_cfg = dict(
            seeds=[85, 100, 75, 458, 1500],
            z_dim=self.z_dim,
        )
        # 默认是4。如果报错“OSError: [WinError 1455] 页面文件太小,无法完成操作”，设置为2或0解决。
        self.data_num_workers = 2

    def get_model(self, device, batch_size=1):
        from mmgan.models import StyleGANv3_SynthesisNetwork, StyleGANv3_MappingNetwork, StyleGANv3_Discriminator, StyleGANv2ADA_AugmentPipe
        from mmgan.models import StyleGANv3Model
        if getattr(self, "model", None) is None:
            # 修改配置。
            self.model_cfg['ema_kimg'] = batch_size * 10 / 32
            self.synthesis['magnitude_ema_beta'] = 0.5 ** (batch_size / (20 * 1e3))
            if self.stylegan_cfg == 'stylegan3-r':
                self.synthesis['conv_kernel'] = 1
                self.synthesis['channel_base'] *= 2
                self.synthesis['channel_max'] *= 2
                self.synthesis['use_radial_filters'] = True
                self.model_cfg['blur_init_sigma'] = 10
                self.model_cfg['blur_fade_kimg'] = batch_size * 200 / 32

            synthesis = StyleGANv3_SynthesisNetwork(**self.synthesis)
            synthesis_ema = StyleGANv3_SynthesisNetwork(**self.synthesis)
            self.mapping['num_ws'] = synthesis.num_ws
            mapping = StyleGANv3_MappingNetwork(**self.mapping)
            mapping_ema = StyleGANv3_MappingNetwork(**self.mapping)
            for name, param in synthesis_ema.named_parameters():
                param.requires_grad = False
            for name, param in mapping_ema.named_parameters():
                param.requires_grad = False
            discriminator = StyleGANv3_Discriminator(**self.discriminator)
            augment_pipe = None
            adjust_p = False  # 是否调整augment_pipe的p
            if hasattr(self, 'augment_pipe') and (self.model_cfg['augment_p'] > 0 or self.model_cfg['ada_target'] is not None):
                augment_pipe = StyleGANv2ADA_AugmentPipe(**self.augment_pipe).train().requires_grad_(False)
                augment_pipe.p.copy_(torch.as_tensor(self.model_cfg['augment_p']))
                if self.model_cfg['ada_target'] is not None:
                    adjust_p = True

            # 第三方实现stylegan3时，不要忘记创建G和D的实例时，都需要设置其的requires_grad_(False)，因为第0步训练Gmain阶段时，D的权重应该不允许得到梯度。
            synthesis.requires_grad_(False)
            synthesis_ema.requires_grad_(False)
            mapping.requires_grad_(False)
            mapping_ema.requires_grad_(False)
            discriminator.requires_grad_(False)
            self.model = StyleGANv3Model(synthesis, synthesis_ema, mapping, mapping_ema,
                                         discriminator=discriminator, augment_pipe=augment_pipe, adjust_p=adjust_p, **self.model_cfg)
        return self.model

    def get_data_loader(
        self, batch_size, is_distributed, cache_img=False
    ):
        from mmgan.data import (
            StyleGANv2ADADataset,
            InfiniteSampler,
            worker_init_reset_seed,
        )
        from mmgan.utils import (
            wait_for_the_master,
            get_local_rank,
        )

        local_rank = get_local_rank()

        with wait_for_the_master(local_rank):
            train_dataset = StyleGANv2ADADataset(
                dataroot=self.dataroot,
                batch_size=batch_size,
                **self.dataset_train_cfg,
            )

        self.dataset = train_dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), shuffle=False, seed=self.seed if self.seed else 0)

        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=True,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        # collater = PPYOLOTrainCollater(self.context, batch_transforms, self.n_layers)
        # dataloader_kwargs["collate_fn"] = collater
        train_loader = torch.utils.data.DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        return 1

    def preprocess(self, inputs, targets, tsize):
        return 1

    def get_optimizer(self, lr, name):
        if name == 'G':
            if "optimizer_G" not in self.__dict__:
                # lr = 0.1   # 用于梯度对齐时换为SGD优化器时解除注释
                param_groups = []
                for name, param in self.model.synthesis.named_parameters():
                    freeze = False
                    for freeze_name in self.synthesis_freeze_at:
                        if freeze_name in name:
                            freeze = True
                            break
                    if not freeze:
                        params0 = {'params': [param]}
                        params0['lr'] = lr
                        param_groups.append(params0)
                    else:
                        param.requires_grad = False
                for name, param in self.model.mapping.named_parameters():
                    params0 = {'params': [param]}
                    params0['lr'] = lr
                    param_groups.append(params0)
                optimizer = torch.optim.Adam(
                    param_groups, lr=lr,
                    betas=(self.optimizer_cfg['generator']['beta1'], self.optimizer_cfg['generator']['beta2']),
                    eps=self.optimizer_cfg['generator']['epsilon']
                )
                # optimizer = torch.optim.SGD(
                #     param_groups, lr=lr, momentum=0.9
                # )
                self.optimizer_G = optimizer
            return self.optimizer_G
        elif name == 'D':
            if "optimizer_D" not in self.__dict__:
                # lr = 0.2   # 用于梯度对齐时换为SGD优化器时解除注释
                param_groups = []
                for name, param in self.model.discriminator.named_parameters():
                    freeze = False
                    for freeze_name in self.discriminator_freeze_at:
                        if freeze_name in name:
                            freeze = True
                            break
                    if not freeze:
                        params0 = {'params': [param]}
                        params0['lr'] = lr
                        param_groups.append(params0)
                    else:
                        param.requires_grad = False
                optimizer = torch.optim.Adam(
                    param_groups, lr=lr,
                    betas=(self.optimizer_cfg['discriminator']['beta1'], self.optimizer_cfg['discriminator']['beta2']),
                    eps=self.optimizer_cfg['discriminator']['epsilon']
                )
                # optimizer = torch.optim.SGD(
                #     param_groups, lr=lr, momentum=0.9
                # )
                self.optimizer_D = optimizer
            return self.optimizer_D

    def get_lr_scheduler(self, lr, iters_per_epoch):
        pass

    def get_eval_loader(self, batch_size, is_distributed):
        from mmgan.data import StyleGANv2ADATestDataset
        val_dataset = StyleGANv2ADATestDataset(**self.dataset_test_cfg)

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(val_dataset)

        dataloader_kwargs = {
            "num_workers": 0,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(val_dataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        pass

    def eval(self, model, evaluator, is_distributed, half=False):
        pass
