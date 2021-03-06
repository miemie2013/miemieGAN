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

        self.kimgs = 25000
        self.basic_glr_per_img = 0.0025 / 16.0
        self.basic_dlr_per_img = 0.0025 / 16.0
        self.synthesis_freeze_at = []
        # self.synthesis_freeze_at = ['b8', 'b16', 'b32', 'b64', 'b128']
        self.discriminator_freeze_at = []
        # self.discriminator_freeze_at = ['b8', 'b16', 'b32', 'b64', 'b128']
        # self.dataroot = '../data/data42681/afhq/train/dog'
        self.dataroot = '../data/flowers_512'

        # 判断是否是调试状态
        isDebug = True if sys.gettrace() else False
        if isDebug:
            print('Debug Mode.')
            self.dataroot = '../' + self.dataroot
            self.output_dir = '../' + self.output_dir
