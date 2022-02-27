#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @miemie2013

import os
import sys

from mmgan.exp import StyleGANv2ADA_Method_Exp


class Exp(StyleGANv2ADA_Method_Exp):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.flip_ema = True
        self.kimgs = 300
        self.basic_lr_per_img = 0.0025 / 64.0
        self.synthesis_freeze_at = []
        # self.synthesis_freeze_at = ['b8', 'b16', 'b32', 'b64', 'b128']
        self.discriminator_freeze_at = []
        # self.discriminator_freeze_at = ['b8', 'b16', 'b32', 'b64', 'b128']
        self.dataroot = '../data/data42681/afhq/train/dog'
        self.model_cfg['flip_ema'] = self.flip_ema

        # 判断是否是调试状态
        isDebug = True if sys.gettrace() else False
        if isDebug:
            print('Debug Mode.')
            self.dataroot = '../' + self.dataroot
            self.output_dir = '../' + self.output_dir
