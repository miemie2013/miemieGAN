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



class StyleGANv2ADA_Method_Exp(BaseExp):
    def __init__(self):
        super().__init__()
        # ---------------- architecture name(算法名) ---------------- #
        self.archi_name = 'StyleGANv2ADA'

        # --------------  training config --------------------- #
        self.G_reg_interval = 4
        self.D_reg_interval = 16

        self.max_epoch = None
        self.kimgs = 25000
        # self.kimgs = 30
        self.print_interval = 10
        self.eval_interval = 1
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # learning_rate
        self.scheduler = "warm_piecewisedecay"
        self.warmup_epochs = -1
        # self.basic_lr_per_img = 0.0025 / 64.0
        self.basic_lr_per_img = 0.0025 / 2.0
        self.start_factor = 0.0
        self.decay_gamma = 0.1
        self.milestones_epoch = [99999998, 99999999]
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
        self.output_dir = "StyleGANv2ADA_outputs"
        self.w_dim = 512
        self.z_dim = 512
        self.c_dim = 0
        self.img_resolution = 512
        self.img_channels = 3
        self.channel_base = 32768
        self.channel_max = 512
        self.num_fp16_res = 4
        self.conv_clamp = 256
        self.synthesis_type = 'StyleGANv2ADA_SynthesisNetwork'
        self.synthesis = dict(
            w_dim=self.w_dim,
            img_resolution=self.img_resolution,
            img_channels=self.img_channels,
            channel_base=self.channel_base,
            channel_max=self.channel_max,
            num_fp16_res=self.num_fp16_res,
            conv_clamp=self.conv_clamp,
        )
        self.mapping_type = 'StyleGANv2ADA_MappingNetwork'
        self.mapping = dict(
            z_dim=self.z_dim,
            c_dim=self.c_dim,
            w_dim=self.w_dim,
            num_ws=16,
            num_layers=8,
        )
        self.discriminator_type = 'StyleGANv2ADA_Discriminator'
        self.discriminator = dict(
            c_dim=self.c_dim,
            img_resolution=self.img_resolution,
            img_channels=self.img_channels,
            channel_base=self.channel_base,
            channel_max=self.channel_max,
            num_fp16_res=self.num_fp16_res,
            conv_clamp=self.conv_clamp,
            block_kwargs={},
            mapping_kwargs={},
            epilogue_kwargs=dict(mbstd_group_size=8,),
        )
        self.model_cfg = dict(
            G_reg_interval=self.G_reg_interval,
            D_reg_interval=self.D_reg_interval,
            r1_gamma=0.5,
            pl_batch_shrink=2,  # default is 2. when train batch_size is 1, set to 1.
            ema_kimg=20,
            ema_rampup=None,
        )

        # ---------------- dataset config ---------------- #
        self.dataroot = '../data/data42681/afhq/train/cat'
        # self.dataroot = '../data/data42681/afhq/train/dog'
        self.dataset_train_cfg = dict(
            resolution=self.img_resolution,
            use_labels=False,
            xflip=False,
            len_phases=4,
        )
        self.dataset_test_cfg = dict(
            seeds=[85, 100, 75, 458, 1500],
            z_dim=self.z_dim,
        )
        # 默认是4。如果报错“OSError: [WinError 1455] 页面文件太小,无法完成操作”，设置为2或0解决。
        self.data_num_workers = 2

    def get_model(self):
        from mmgan.models import StyleGANv2ADA_SynthesisNetwork, StyleGANv2ADA_MappingNetwork, StyleGANv2ADA_Discriminator
        from mmgan.models import StyleGANv2ADAModel
        if getattr(self, "model", None) is None:
            synthesis = StyleGANv2ADA_SynthesisNetwork(**self.synthesis)
            synthesis_ema = StyleGANv2ADA_SynthesisNetwork(**self.synthesis)
            mapping = StyleGANv2ADA_MappingNetwork(**self.mapping)
            mapping_ema = StyleGANv2ADA_MappingNetwork(**self.mapping)
            discriminator = StyleGANv2ADA_Discriminator(**self.discriminator)
            augment_pipe = None
            self.model = StyleGANv2ADAModel(synthesis, synthesis_ema, mapping, mapping_ema,
                                            discriminator=discriminator, augment_pipe=augment_pipe, **self.model_cfg)
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
        import itertools
        if name == 'G':
            if "optimizer_G" not in self.__dict__:
                if self.warmup_epochs > 0:
                    lr = lr * self.start_factor

                optimizer = torch.optim.Adam(
                    itertools.chain(self.model.synthesis.parameters(), self.model.mapping.parameters()), lr=lr,
                    betas=(self.optimizer_cfg['generator']['beta1'], self.optimizer_cfg['generator']['beta2']),
                    eps=self.optimizer_cfg['generator']['epsilon']
                )
                self.optimizer_G = optimizer
            return self.optimizer_G
        elif name == 'D':
            if "optimizer_D" not in self.__dict__:
                if self.warmup_epochs > 0:
                    lr = lr * self.start_factor

                optimizer = torch.optim.Adam(
                    self.model.discriminator.parameters(), lr=lr,
                    betas=(self.optimizer_cfg['discriminator']['beta1'], self.optimizer_cfg['discriminator']['beta2']),
                    eps=self.optimizer_cfg['discriminator']['epsilon']
                )
                self.optimizer_D = optimizer
            return self.optimizer_D

    def get_lr_scheduler(self, lr, iters_per_epoch):
        from mmgan.utils import LRScheduler

        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=lr * self.start_factor,
            milestones=self.milestones_epoch,
            gamma=self.decay_gamma,
        )
        return scheduler

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
