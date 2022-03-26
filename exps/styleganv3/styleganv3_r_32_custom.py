#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @miemie2013

import os
import sys

from mmgan.exp import StyleGANv3_Method_Exp


class Exp(StyleGANv3_Method_Exp):
    def __init__(self):
        super(Exp, self).__init__()
        # ---------------- architecture name(算法名) ---------------- #
        self.archi_name = 'StyleGANv3'

        # --------------  training config --------------------- #
        self.G_reg_interval = None
        self.D_reg_interval = 16

        self.max_epoch = None
        self.kimgs = 5000
        self.print_interval = 10
        self.temp_img_interval = 100
        self.eval_interval = 1
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # learning_rate
        self.basic_glr_per_img = 0.0025 / 16.0
        self.basic_dlr_per_img = 0.002 / 16.0
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
        self.img_resolution = 32
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
            r1_gamma=6.6,
            pl_batch_shrink=2,
            ema_kimg=-1,
            ema_rampup=None,
            augment_p=0.0,
            ada_kimg=100,
            ada_interval=4,
            ada_target=0.6,
        )

        # ---------------- dataset config ---------------- #
        self.dataroot = '../data/data42681/afhq/train/dog_32'
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

        # 判断是否是调试状态
        isDebug = True if sys.gettrace() else False
        if isDebug:
            print('Debug Mode.')
            self.dataroot = '../' + self.dataroot
            self.output_dir = '../' + self.output_dir
