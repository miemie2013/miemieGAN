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
        self.save_step_interval = 1000
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
        self.output_dir = "StarGANv2_outputs"
        self.latent_dim = 16
        self.style_dim = 64
        self.img_size = 256
        self.num_domains = 3
        self.lambda_sty = 1
        self.lambda_ds = 2
        self.lambda_cyc = 1
        self.generator_type = 'StarGANv2Generator'
        self.generator = dict(
            img_size=self.img_size,
            w_hpf=0,
            style_dim=self.style_dim,
        )
        self.style_type = 'StarGANv2Style'
        self.style = dict(
            img_size=self.img_size,
            style_dim=self.style_dim,
            num_domains=self.num_domains,
        )
        self.mapping_type = 'StarGANv2Mapping'
        self.mapping = dict(
            latent_dim=self.latent_dim,
            style_dim=self.style_dim,
            num_domains=self.num_domains,
        )
        self.discriminator_type = 'StarGANv2Discriminator'
        self.discriminator = dict(
            img_size=self.img_size,
            num_domains=self.num_domains,
        )
        self.model_cfg = dict(
            latent_dim=self.latent_dim,
            lambda_sty=self.lambda_sty,
            lambda_ds=self.lambda_ds,
            lambda_cyc=self.lambda_cyc,
        )

        # ---------------- dataset config ---------------- #
        self.dataroot = '../data/data42681/afhq/train/cat'
        # self.dataroot = '../data/data42681/afhq/train/dog'
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

    def get_model(self, device, rank, batch_size=1):
        from mmgan.models import StarGANv2Style, StarGANv2Mapping, StarGANv2Generator, StarGANv2Discriminator
        from mmgan.models import StarGANv2Model
        if getattr(self, "model", None) is None:
            generator = StarGANv2Generator(**self.generator).train().requires_grad_(False).to(device)
            generator_ema = StarGANv2Generator(**self.generator).eval().requires_grad_(False).to(device)
            style = StarGANv2Style(**self.style).train().requires_grad_(False).to(device)
            style_ema = StarGANv2Style(**self.style).eval().requires_grad_(False).to(device)
            mapping = StarGANv2Mapping(**self.mapping).train().requires_grad_(False).to(device)
            mapping_ema = StarGANv2Mapping(**self.mapping).eval().requires_grad_(False).to(device)
            discriminator = StarGANv2Discriminator(**self.discriminator).train().requires_grad_(False).to(device)

            fan = None
            fan_ema = None
            if generator.w_hpf > 0:
                # fan = build_generator(fan)
                # fan.eval()
                pass

            generator.requires_grad_(False)
            generator_ema.requires_grad_(False)
            style.requires_grad_(False)
            style_ema.requires_grad_(False)
            mapping.requires_grad_(False)
            mapping_ema.requires_grad_(False)
            discriminator.requires_grad_(False)
            self.model = StarGANv2Model(generator, generator_ema, style, style_ema, mapping, mapping_ema,
                                        fan, fan_ema, discriminator, device, rank, **self.model_cfg)
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

        sampler = InfiniteSampler(len(self.dataset), shuffle=True, seed=self.seed if self.seed else 0)

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
