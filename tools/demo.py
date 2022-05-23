#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @miemie2013

import argparse
import os
import time

import numpy as np
from loguru import logger

import cv2
import copy
import imageio
import PIL.Image
from time import perf_counter
import torch
import torch.nn.functional as F

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
        "demo", default="image", help="demo type, eg. image, style_mixing, A2B, projector"
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
    parser.add_argument(
        "--projector_random_seed",
        default=303,
        type=int,
        help="projector_random_seed",
    )
    parser.add_argument(
        "--target_fname",
        default='',
        type=str,
        help="projector target_fname",
    )
    parser.add_argument(
        "--num_steps",
        default=1000,
        type=int,
        help="projector num_steps",
    )
    parser.add_argument(
        "--outdir",
        default='',
        type=str,
        help="projector outdir",
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


import io
import re
import requests
import html
import hashlib
import glob
import urllib
import urllib.request
import uuid
import tempfile
from typing import Any, List, Tuple, Union


def is_url(obj: Any, allow_file_urls: bool = False) -> bool:
    """Determine whether the given object is a valid URL string."""
    if not isinstance(obj, str) or not "://" in obj:
        return False
    if allow_file_urls and obj.startswith('file://'):
        return True
    try:
        res = requests.compat.urlparse(obj)
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
        res = requests.compat.urlparse(requests.compat.urljoin(obj, "/"))
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
    except:
        return False
    return True


_dnnlib_cache_dir = None

def make_cache_dir_path(*paths: str) -> str:
    if _dnnlib_cache_dir is not None:
        return os.path.join(_dnnlib_cache_dir, *paths)
    if 'DNNLIB_CACHE_DIR' in os.environ:
        return os.path.join(os.environ['DNNLIB_CACHE_DIR'], *paths)
    if 'HOME' in os.environ:
        return os.path.join(os.environ['HOME'], '.cache', 'dnnlib', *paths)
    if 'USERPROFILE' in os.environ:
        return os.path.join(os.environ['USERPROFILE'], '.cache', 'dnnlib', *paths)
    return os.path.join(tempfile.gettempdir(), '.cache', 'dnnlib', *paths)


def open_url(url: str, cache_dir: str = None, num_attempts: int = 10, verbose: bool = True, return_filename: bool = False, cache: bool = True) -> Any:
    """Download the given URL and return a binary-mode file object to access the data."""
    assert num_attempts >= 1
    assert not (return_filename and (not cache))

    # Doesn't look like an URL scheme so interpret it as a local filename.
    if not re.match('^[a-z]+://', url):
        return url if return_filename else open(url, "rb")

    # Handle file URLs.  This code handles unusual file:// patterns that
    # arise on Windows:
    #
    # file:///c:/foo.txt
    #
    # which would translate to a local '/c:/foo.txt' filename that's
    # invalid.  Drop the forward slash for such pathnames.
    #
    # If you touch this code path, you should test it on both Linux and
    # Windows.
    #
    # Some internet resources suggest using urllib.request.url2pathname() but
    # but that converts forward slashes to backslashes and this causes
    # its own set of problems.
    if url.startswith('file://'):
        filename = urllib.parse.urlparse(url).path
        if re.match(r'^/[a-zA-Z]:', filename):
            filename = filename[1:]
        return filename if return_filename else open(filename, "rb")

    assert is_url(url)

    # Lookup from cache.
    if cache_dir is None:
        cache_dir = make_cache_dir_path('downloads')

    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()
    if cache:
        cache_files = glob.glob(os.path.join(cache_dir, url_md5 + "_*"))
        if len(cache_files) == 1:
            filename = cache_files[0]
            return filename if return_filename else open(filename, "rb")

    # Download.
    url_name = None
    url_data = None
    with requests.Session() as session:
        if verbose:
            print("Downloading %s ..." % url, end="", flush=True)
        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    if len(res.content) == 0:
                        raise IOError("No data received")

                    if len(res.content) < 8192:
                        content_str = res.content.decode("utf-8")
                        if "download_warning" in res.headers.get("Set-Cookie", ""):
                            links = [html.unescape(link) for link in content_str.split('"') if "export=download" in link]
                            if len(links) == 1:
                                url = requests.compat.urljoin(url, links[0])
                                raise IOError("Google Drive virus checker nag")
                        if "Google Drive - Quota exceeded" in content_str:
                            raise IOError("Google Drive download quota exceeded -- please try again later")

                    match = re.search(r'filename="([^"]*)"', res.headers.get("Content-Disposition", ""))
                    url_name = match[1] if match else url
                    url_data = res.content
                    if verbose:
                        print(" done")
                    break
            except KeyboardInterrupt:
                raise
            except:
                if not attempts_left:
                    if verbose:
                        print(" failed")
                    raise
                if verbose:
                    print(".", end="", flush=True)

    # Save to cache.
    if cache:
        safe_name = re.sub(r"[^0-9a-zA-Z-._]", "_", url_name)
        cache_file = os.path.join(cache_dir, url_md5 + "_" + safe_name)
        temp_file = os.path.join(cache_dir, "tmp_" + uuid.uuid4().hex + "_" + url_md5 + "_" + safe_name)
        os.makedirs(cache_dir, exist_ok=True)
        with open(temp_file, "wb") as f:
            f.write(url_data)
        os.replace(temp_file, cache_file) # atomic
        if return_filename:
            return cache_file

    # Return data as file object.
    assert not return_filename
    return io.BytesIO(url_data)


def project(
    mapping_ema,
    synthesis_ema,
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    *,
    num_steps                  = 1000,
    w_avg_samples              = 10000,
    initial_learning_rate      = 0.1,
    initial_noise_factor       = 0.05,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.75,
    regularize_noise_weight    = 1e5,
    verbose                    = False,
    device: torch.device
):
    assert target.shape == (synthesis_ema.img_channels, synthesis_ema.img_resolution, synthesis_ema.img_resolution)

    def logprint(*args):
        if verbose:
            logger.info(*args)

    mapping_ema = copy.deepcopy(mapping_ema).eval().requires_grad_(False).to(device)
    synthesis_ema = copy.deepcopy(synthesis_ema).eval().requires_grad_(False).to(device)

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, mapping_ema.z_dim)
    w_samples = mapping_ema(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in synthesis_ema.named_buffers() if 'noise_const' in name }

    # Load VGG16 feature detector.
    # url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    # with dnnlib.util.open_url(url) as f:
    #     vgg16 = torch.jit.load(f).eval().to(device)
    url = 'vgg16.pt'
    with open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, synthesis_ema.num_ws, 1])
        synth_images = synthesis_ema(ws, noise_mode='const')

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255/2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step+1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_out.repeat([1, synthesis_ema.num_ws, 1])


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
                    assert len(A2B_mixing_seed) == 1, 'A2B_mixing_seed only supports 1 seed.'
                    z = np.random.RandomState(A2B_mixing_seed[0]).randn(1, model.z_dim)
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
    elif args.demo == "projector":   # 投影仪
        # 不同的算法输入不同，新增算法时这里也要增加elif
        if archi_name == 'StyleGANv2ADA' or archi_name == 'StyleGANv3':
            projector_random_seed = args.projector_random_seed
            np.random.seed(projector_random_seed)
            torch.manual_seed(projector_random_seed)

            model.synthesis_ema.requires_grad_(False)
            model.mapping_ema.requires_grad_(False)

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

            synthesis_ema = model.synthesis_ema
            mapping_ema = model.mapping_ema

            # 加载目标真实图片
            target_pil = PIL.Image.open(args.target_fname).convert('RGB')
            w, h = target_pil.size
            s = min(w, h)
            target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
            target_pil = target_pil.resize((synthesis_ema.img_resolution, synthesis_ema.img_resolution), PIL.Image.LANCZOS)
            target_uint8 = np.array(target_pil, dtype=np.uint8)

            # 优化投影
            start_time = perf_counter()
            projected_w_steps = project(
                mapping_ema, synthesis_ema,
                target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
                num_steps=args.num_steps,
                device=device,
                verbose=True
            )
            logger.info(f'Elapsed: {(perf_counter()-start_time):.1f} s')


            # Render debug output: optional video and projected image and W vector.
            outdir = args.outdir
            os.makedirs(outdir, exist_ok=True)
            save_video = True
            if save_video:
                video = imageio.get_writer(f'{outdir}/proj.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
                print (f'Saving optimization progress video "{outdir}/proj.mp4"')
                for projected_w in projected_w_steps:
                    synth_image = synthesis_ema(projected_w.unsqueeze(0), noise_mode='const')
                    synth_image = (synth_image + 1) * (255/2)
                    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                    video.append_data(np.concatenate([target_uint8, synth_image], axis=1))
                video.close()

            # Save final projected frame and W vector.
            target_pil.save(f'{outdir}/target.png')
            projected_w = projected_w_steps[-1]
            synth_image = synthesis_ema(projected_w.unsqueeze(0), noise_mode='const')
            synth_image = (synth_image + 1) * (255/2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/proj.png')
            np.savez(f'{outdir}/projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())
            logger.info("Saving projected_w result in {}".format(f'{outdir}/projected_w.npz'))
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
