#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @miemie2013

import os
import sys

from mmgan.exp import StyleGANv2ADA_Method_Exp


class Exp(StyleGANv2ADA_Method_Exp):
    def __init__(self):
        super(Exp, self).__init__()
        # ---------------- architecture name(算法名) ---------------- #
        self.archi_name = 'StyleGANv2ADA'

        # --------------  training config --------------------- #
        self.G_reg_interval = 4
        self.D_reg_interval = 16

        self.max_epoch = None
        self.kimgs = 25000
        self.print_interval = 10
        self.temp_img_interval = 100
        self.eval_interval = 1
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # learning_rate
        # self.basic_lr_per_img = 0.0025 / 64.0
        self.basic_lr_per_img = 0.0025 / 16.0
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
        self.img_resolution = 256
        self.img_channels = 3
        self.channel_base = 32768
        self.channel_max = 512
        self.num_fp16_res = 4
        self.conv_clamp = 256
        self.synthesis_freeze_at = []
        # self.synthesis_freeze_at = ['b8', 'b16', 'b32', 'b64', 'b128']
        self.discriminator_freeze_at = []
        # self.discriminator_freeze_at = ['b8', 'b16', 'b32', 'b64', 'b128']
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
            epilogue_kwargs=dict(mbstd_group_size=None,),   # mbstd_group_size要从8改成None，使得单卡支持的最大批大小大于8。
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
            r1_gamma=0.5,
            pl_batch_shrink=2,  # default is 2. when train batch_size is 1, set to 1.
            ema_kimg=20,
            ema_rampup=None,
            augment_p=0.0,
            ada_kimg=100,
            ada_interval=4,
            ada_target=0.6,
        )

        # ---------------- dataset config ---------------- #
        self.dataroot = '../data/data110820/faces'
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

        # 判断是否是调试状态
        isDebug = True if sys.gettrace() else False
        if isDebug:
            print('Debug Mode.')
            self.dataroot = '../' + self.dataroot
            self.output_dir = '../' + self.output_dir
