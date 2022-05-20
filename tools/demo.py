#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @miemie2013

import argparse
import os
import time

import numpy as np
from loguru import logger

import cv2

import torch

# add python path of this repo to sys.path
import sys
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from mmgan.data.data_augment import *
from mmgan.exp import get_exp
from mmgan.utils import fuse_model, get_model_info, postprocess, vis, get_classes, vis2, load_ckpt

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("MieMieGAN Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, style_mixing, A2B"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)

    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    parser.add_argument(
        "--seeds",
        default="85,100,75,458,1500",
        type=str,
        help="random seeds",
    )
    parser.add_argument(
        "--row_seeds",
        default="85,100,75,458,1500",
        type=str,
        help="random seeds",
    )
    parser.add_argument(
        "--col_seeds",
        default="55,821,1789,293",
        type=str,
        help="random seeds",
    )
    parser.add_argument(
        "--col_styles",
        default="0,1,2,3,4,5,6",
        type=str,
        help="col_styles, for example you can set col_styles to \'0,1,2,3,4,5,6\'",
    )
    parser.add_argument(
        "--frames",
        default=120,
        type=int,
        help="A2B frames",
    )
    parser.add_argument(
        "--video_fps",
        default=30,
        type=int,
        help="A2B video_fps",
    )
    parser.add_argument(
        "--A2B_mixing_seed",
        default="",
        type=str,
        help="A2B mixing_seed",
    )
    parser.add_argument(
        "--noise_mode",
        default="const",
        type=str,
        help="noise_mode, assert noise_mode in ['random', 'const', 'none']",
    )
    parser.add_argument(
        "--trunc",
        default=1.0,
        type=float,
        help="truncation_psi",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    return parser


def get_seeds(seeds):
    if ',' in seeds:
        seeds = seeds.split(',')
    elif '_' in seeds:
        seeds_start_end = seeds.split('_')
        seeds_start = int(seeds_start_end[0])
        seeds_end = int(seeds_start_end[1])
        assert seeds_start < seeds_end
        seeds = []
        for ii in range(seeds_start, seeds_end + 1, 1):
            seeds.append(ii)
    elif seeds.isdigit():
        seeds = [int(seeds)]
    else:
        raise NotImplementedError("seeds \'{}\' can not be analyzed.".format(seeds))
    seeds = [int(seed) for seed in seeds]
    return seeds



def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    logger.info("Args: {}".format(args))

    # 算法名字
    archi_name = exp.archi_name

    device = torch.device('cpu')
    if args.device == "gpu":
        device = torch.device('cuda:0')

    if archi_name == 'StyleGANv2ADA':
        model = exp.get_model(device, 0)
        # 方便style_mixing输入合理的col_styles
        logger.info('num_ws = %d'%model.synthesis_ema.num_ws)
    elif archi_name == 'StyleGANv3':
        model = exp.get_model(device, 0, batch_size=1)
        # 方便style_mixing输入合理的col_styles
        logger.info('num_ws = %d'%model.synthesis_ema.num_ws)
    else:
        raise NotImplementedError("Architectures \'{}\' is not implemented.".format(archi_name))


    if args.device == "gpu":
        model.synthesis.cuda()
        model.synthesis_ema.cuda()
        model.mapping.cuda()
        model.mapping_ema.cuda()
        model.discriminator.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.synthesis.eval()
    model.synthesis_ema.eval()
    model.mapping.eval()
    model.mapping_ema.eval()
    model.discriminator.eval()

    if args.demo == "image":
        # 不同的算法输入不同，新增算法时这里也要增加elif
        if archi_name == 'StyleGANv2ADA' or archi_name == 'StyleGANv3':
            assert args.noise_mode in ['random', 'const', 'none']
            assert 0.0 <= args.trunc <= 1.0
            # 加载模型权重
            if args.ckpt is None:
                ckpt_file = os.path.join(file_name, "best_ckpt.pth")
            else:
                ckpt_file = args.ckpt
            logger.info("loading checkpoint")
            ckpt = torch.load(ckpt_file, map_location="cpu")
            model.synthesis = load_ckpt(model.synthesis, ckpt["synthesis"])
            model.synthesis_ema = load_ckpt(model.synthesis_ema, ckpt["synthesis_ema"])
            model.mapping = load_ckpt(model.mapping, ckpt["mapping"])
            model.mapping_ema = load_ckpt(model.mapping_ema, ckpt["mapping_ema"])
            model.discriminator = load_ckpt(model.discriminator, ckpt["discriminator"])
            logger.info("loaded checkpoint done.")

            seeds = get_seeds(args.seeds)
            current_time = time.localtime()

            for seed in seeds:
                z = np.random.RandomState(seed).randn(1, model.z_dim)
                z = torch.from_numpy(z)
                seed = np.array([seed]).astype(np.int32)
                seed = torch.from_numpy(seed)
                z = z.float()
                if args.device == "gpu":
                    z = z.cuda()
                    if args.fp16:
                        z = z.half()  # to FP16
                data = {
                    'z': z,
                    'seed': seed,
                }
                model.setup_input(data)
                with torch.no_grad():
                    img_bgr, seed_i = model.test_iter(noise_mode=args.noise_mode, truncation_psi=args.trunc)
                    if args.save_result:
                        save_folder = os.path.join(
                            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
                        )
                        os.makedirs(save_folder, exist_ok=True)
                        save_file_name = os.path.join(save_folder, f'seed{seed_i:08d}.png')
                        logger.info("Saving generation result in {}".format(save_file_name))
                        cv2.imwrite(save_file_name, img_bgr)

        else:
            raise NotImplementedError("Architectures \'{}\' is not implemented.".format(archi_name))
    elif args.demo == "style_mixing":
        # 不同的算法输入不同，新增算法时这里也要增加elif
        if archi_name == 'StyleGANv2ADA' or archi_name == 'StyleGANv3':
            # 加载模型权重
            if args.ckpt is None:
                ckpt_file = os.path.join(file_name, "best_ckpt.pth")
            else:
                ckpt_file = args.ckpt
            logger.info("loading checkpoint")
            ckpt = torch.load(ckpt_file, map_location="cpu")
            model.synthesis = load_ckpt(model.synthesis, ckpt["synthesis"])
            model.synthesis_ema = load_ckpt(model.synthesis_ema, ckpt["synthesis_ema"])
            model.mapping = load_ckpt(model.mapping, ckpt["mapping"])
            model.mapping_ema = load_ckpt(model.mapping_ema, ckpt["mapping_ema"])
            model.discriminator = load_ckpt(model.discriminator, ckpt["discriminator"])
            logger.info("loaded checkpoint done.")

            row_seeds = get_seeds(args.row_seeds)
            col_seeds = get_seeds(args.col_seeds)
            col_styles = get_seeds(args.col_styles)
            all_seeds = list(set(row_seeds + col_seeds))
            current_time = time.localtime()

            all_z = np.stack([np.random.RandomState(seed).randn(model.z_dim) for seed in all_seeds])
            all_z = torch.from_numpy(all_z)
            all_z = all_z.float()
            if args.device == "gpu":
                all_z = all_z.cuda()
                if args.fp16:
                    all_z = all_z.half()  # to FP16
            data = {
                'z': all_z,
            }
            model.setup_input(data)
            with torch.no_grad():
                img_bgr = model.style_mixing(row_seeds, col_seeds, all_seeds, col_styles, noise_mode=args.noise_mode, truncation_psi=args.trunc)
                if args.save_result:
                    save_folder = os.path.join(
                        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
                    )
                    os.makedirs(save_folder, exist_ok=True)
                    save_file_name = os.path.join(save_folder, f'style_mixing.png')
                    logger.info("Saving generation result in {}".format(save_file_name))
                    cv2.imwrite(save_file_name, img_bgr)

        else:
            raise NotImplementedError("Architectures \'{}\' is not implemented.".format(archi_name))
    elif args.demo == "A2B":
        # 不同的算法输入不同，新增算法时这里也要增加elif
        if archi_name == 'StyleGANv2ADA' or archi_name == 'StyleGANv3':
            # 加载模型权重
            if args.ckpt is None:
                ckpt_file = os.path.join(file_name, "best_ckpt.pth")
            else:
                ckpt_file = args.ckpt
            logger.info("loading checkpoint")
            ckpt = torch.load(ckpt_file, map_location="cpu")
            model.synthesis = load_ckpt(model.synthesis, ckpt["synthesis"])
            model.synthesis_ema = load_ckpt(model.synthesis_ema, ckpt["synthesis_ema"])
            model.mapping = load_ckpt(model.mapping, ckpt["mapping"])
            model.mapping_ema = load_ckpt(model.mapping_ema, ckpt["mapping_ema"])
            model.discriminator = load_ckpt(model.discriminator, ckpt["discriminator"])
            logger.info("loaded checkpoint done.")

            seeds = get_seeds(args.seeds)
            A2B_mixing_seed = args.A2B_mixing_seed
            A2B_mixing_w = None
            col_styles = None
            if A2B_mixing_seed != "":
                if A2B_mixing_seed == "w_avg":
                    # 直接用w_avg妈妈进行渐变的style_mixing
                    A2B_mixing_w = model.mapping_ema.w_avg
                    A2B_mixing_w = A2B_mixing_w.unsqueeze(0).unsqueeze(0).repeat([1, model.synthesis_ema.num_ws, 1])
                else:
                    A2B_mixing_seed = get_seeds(A2B_mixing_seed)
                    z = np.random.RandomState(A2B_mixing_seed).randn(1, model.z_dim)
                    z = torch.from_numpy(z)
                    z = z.float()
                    if args.device == "gpu":
                        z = z.cuda()
                        if args.fp16:
                            z = z.half()  # to FP16
                    seed = np.array([0]).astype(np.int32)
                    seed = torch.from_numpy(seed)
                    data = {
                        'z': z,
                        'seed': seed,
                    }
                    model.setup_input(data)
                    with torch.no_grad():
                        w = model.test_iter(noise_mode=args.noise_mode, truncation_psi=args.trunc, return_ws=True)
                    A2B_mixing_w = w
                # col_styles和A2B_mixing_seed配合使用，有A2B_mixing_seed时才会使用col_styles
                col_styles = get_seeds(args.col_styles)
            current_time = time.localtime()

            ws = []
            for seed in seeds:
                z = np.random.RandomState(seed).randn(1, model.z_dim)
                z = torch.from_numpy(z)
                z = z.float()
                if args.device == "gpu":
                    z = z.cuda()
                    if args.fp16:
                        z = z.half()  # to FP16
                seed = np.array([0]).astype(np.int32)
                seed = torch.from_numpy(seed)
                data = {
                    'z': z,
                    'seed': seed,
                }
                model.setup_input(data)
                with torch.no_grad():
                    w = model.test_iter(noise_mode=args.noise_mode, truncation_psi=args.trunc, return_ws=True)
                ws.append(w)

            total_frames = args.frames * (len(seeds) - 1) + 1
            save_file_names = []
            save_folder = None
            for frame_id in range(total_frames):
                w_idx = frame_id // args.frames
                if frame_id < total_frames - 1:
                    w0 = ws[w_idx]
                    w1 = ws[w_idx + 1]
                    # 插值
                    beta = (frame_id % args.frames) / args.frames
                    w = w0.lerp(w1, beta)   # w1占的比重是beta, w0占的比重是(1.0 - beta)
                else:
                    w = ws[w_idx]
                if A2B_mixing_w is not None:
                    # w = w0.lerp(w1, beta)   # w1占的比重是beta, w0占的比重是(1.0 - beta)
                    w[0][col_styles] = A2B_mixing_w[0][col_styles]
                with torch.no_grad():
                    img_bgr, _ = model.run_synthesis_ema(w, 0, noise_mode=args.noise_mode)
                    if args.save_result:
                        save_folder = os.path.join(
                            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
                        )
                        os.makedirs(save_folder, exist_ok=True)
                        save_file_name = os.path.join(save_folder, f'frame_{frame_id:08d}.png')
                        logger.info("Saving generation result in {}".format(save_file_name))
                        cv2.imwrite(save_file_name, img_bgr)
                        save_file_names.append(save_file_name)
            # 合成视频
            save_video_path = os.path.join(save_folder, f'video.avi')
            frame_0 = cv2.imread(save_file_names[0])
            img_size = (frame_0.shape[0], frame_0.shape[1])
            fps = args.video_fps
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            video_writer = cv2.VideoWriter(save_video_path, fourcc, fps, img_size)
            for fanme in save_file_names:
                frame = cv2.imread(fanme)
                video_writer.write(frame)
            video_writer.release()
            logger.info("Saving video result in {}".format(save_video_path))
        else:
            raise NotImplementedError("Architectures \'{}\' is not implemented.".format(archi_name))


if __name__ == "__main__":
    args = make_parser().parse_args()
    # 判断是否是调试状态
    isDebug = True if sys.gettrace() else False
    if isDebug:
        print('Debug Mode.')
        args.exp_file = '../' + args.exp_file
        args.ckpt = '../' + args.ckpt   # 如果是绝对路径，把这一行注释掉
    exp = get_exp(args.exp_file)

    main(exp, args)
