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

        # 判断是否是调试状态
        isDebug = True if sys.gettrace() else False
        if isDebug:
            print('Debug Mode.')
            self.dataroot = '../' + self.dataroot
            self.output_dir = '../' + self.output_dir
