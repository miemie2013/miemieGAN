#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @miemie2013

import argparse
import os
import time
from loguru import logger

import cv2
import torch
# import paddle.fluid as fluid
import pickle
import six

# add python path of this repo to sys.path
import sys
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from mmgan.exp import get_exp
from mmgan.utils import fuse_model, get_model_info, postprocess, vis, get_classes
from mmgan.models import *
from mmgan.models.custom_layers import *


def make_parser():
    parser = argparse.ArgumentParser("MieMieGAN convert weights")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your experiment description file",
    )
    parser.add_argument("-c_G", "--c_G", default=None, type=str, help="generator checkpoint")
    parser.add_argument("-c_Gema", "--c_Gema", default=None, type=str, help="generator_ema checkpoint")
    parser.add_argument("-c_D", "--c_D", default=None, type=str, help="discriminator checkpoint")
    parser.add_argument("-oc", "--output_ckpt", default=None, type=str, help="output checkpoint")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    return parser


def copy_conv_bn(conv_unit, w, scale, offset, m, v, use_gpu):
    if use_gpu:
        conv_unit.conv.weight.data = torch.Tensor(w).cuda()
        conv_unit.bn.weight.data = torch.Tensor(scale).cuda()
        conv_unit.bn.bias.data = torch.Tensor(offset).cuda()
        conv_unit.bn.running_mean.data = torch.Tensor(m).cuda()
        conv_unit.bn.running_var.data = torch.Tensor(v).cuda()
    else:
        conv_unit.conv.weight.data = torch.Tensor(w)
        conv_unit.bn.weight.data = torch.Tensor(scale)
        conv_unit.bn.bias.data = torch.Tensor(offset)
        conv_unit.bn.running_mean.data = torch.Tensor(m)
        conv_unit.bn.running_var.data = torch.Tensor(v)


def copy_conv_gn(conv_unit, w, b, scale, offset, use_gpu):
    if use_gpu:
        conv_unit.conv.weight.data = torch.Tensor(w).cuda()
        conv_unit.conv.bias.data = torch.Tensor(b).cuda()
        conv_unit.gn.weight.data = torch.Tensor(scale).cuda()
        conv_unit.gn.bias.data = torch.Tensor(offset).cuda()
    else:
        conv_unit.conv.weight.data = torch.Tensor(w)
        conv_unit.conv.bias.data = torch.Tensor(b)
        conv_unit.gn.weight.data = torch.Tensor(scale)
        conv_unit.gn.bias.data = torch.Tensor(offset)

def copy_conv_af(conv_unit, w, scale, offset, use_gpu):
    if use_gpu:
        conv_unit.conv.weight.data = torch.Tensor(w).cuda()
        conv_unit.af.weight.data = torch.Tensor(scale).cuda()
        conv_unit.af.bias.data = torch.Tensor(offset).cuda()
    else:
        conv_unit.conv.weight.data = torch.Tensor(w)
        conv_unit.af.weight.data = torch.Tensor(scale)
        conv_unit.af.bias.data = torch.Tensor(offset)


def copy_conv(conv_layer, w, b, use_gpu):
    if use_gpu:
        conv_layer.weight.data = torch.Tensor(w).cuda()
        conv_layer.bias.data = torch.Tensor(b).cuda()
    else:
        conv_layer.weight.data = torch.Tensor(w)
        conv_layer.bias.data = torch.Tensor(b)


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    logger.info("Args: {}".format(args))

    model = exp.get_model()
    # 算法名字
    model_class_name = model.__class__.__name__

    use_gpu = False
    if args.device == "gpu":
        model.cuda()
        use_gpu = True
    model.eval()

    # 新增算法时这里也要增加elif
    if model_class_name == 'StyleGANv2ADAModel':
        generator_dic = torch.load(args.c_G, map_location=torch.device('cpu'))
        generator_ema_dic = torch.load(args.c_Gema, map_location=torch.device('cpu'))
        discriminator_dic = torch.load(args.c_D, map_location=torch.device('cpu'))

        synthesis = model.synthesis
        synthesis_ema = model.synthesis_ema
        mapping = model.mapping
        mapping_ema = model.mapping_ema
        discriminator = model.discriminator

        synthesis_std = synthesis.state_dict()
        synthesis_ema_std = synthesis_ema.state_dict()
        mapping_std = mapping.state_dict()
        mapping_ema_std = mapping_ema.state_dict()
        # discriminator_std = discriminator.state_dict()

        discriminator.load_state_dict(discriminator_dic)

        for key, value in synthesis_std.items():
            key2 = 'synthesis.' + key
            synthesis_std[key] = generator_dic[key2]
        for key, value in synthesis_ema_std.items():
            key2 = 'synthesis.' + key
            synthesis_ema_std[key] = generator_ema_dic[key2]

        for key, value in mapping_std.items():
            key2 = 'mapping.' + key
            mapping_std[key] = generator_dic[key2]
        for key, value in mapping_ema_std.items():
            key2 = 'mapping.' + key
            mapping_ema_std[key] = generator_ema_dic[key2]

        synthesis.load_state_dict(synthesis_std)
        synthesis_ema.load_state_dict(synthesis_ema_std)
        mapping.load_state_dict(mapping_std)
        mapping_ema.load_state_dict(mapping_ema_std)
    else:
        raise NotImplementedError("Architectures \'{}\' is not implemented.".format(model_class_name))

    # save checkpoint.
    ckpt_state = {
        "start_epoch": 0,
        "model": model.state_dict(),
        "optimizer": None,
    }
    torch.save(ckpt_state, args.output_ckpt)
    logger.info("Done.")


if __name__ == "__main__":
    args = make_parser().parse_args()
    # 判断是否是调试状态
    isDebug = True if sys.gettrace() else False
    if isDebug:
        print('Debug Mode.')
        args.exp_file = '../' + args.exp_file
        args.c_G = '../' + args.c_G
        args.c_Gema = '../' + args.c_Gema
        args.c_D = '../' + args.c_D
        args.output_ckpt = '../' + args.output_ckpt
    exp = get_exp(args.exp_file)

    main(exp, args)
