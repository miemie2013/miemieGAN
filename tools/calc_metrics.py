#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @miemie2013

import argparse
import os
import time
import pickle
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
        "demo", default="fid50k_full", help="demo type, eg. fid50k_full, fid50k_full or fid50k_full"
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
        help="col_styles",
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



class FeatureStats:
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append_torch(self, x, num_gpus=1, rank=0):
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        assert 0 <= rank < num_gpus
        if num_gpus > 1:
            ys = []
            for src in range(num_gpus):
                y = x.clone()
                torch.distributed.broadcast(y, src=src)
                ys.append(y)
            x = torch.stack(ys, dim=1).flatten(0, 1) # interleave samples
        self.append(x.cpu().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = dnnlib.EasyDict(pickle.load(f))
        obj = FeatureStats(capture_all=s.capture_all, max_items=s.max_items)
        obj.__dict__.update(s)
        return obj


def compute_feature_stats_for_dataset(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64, data_loader_kwargs=None, max_items=None, **stats_kwargs):
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

    # Try to lookup from cache.
    cache_file = None
    # if opts.cache:
    #     # Choose cache file name.
    #     args = dict(dataset_kwargs=opts.dataset_kwargs, detector_url=detector_url, detector_kwargs=detector_kwargs, stats_kwargs=stats_kwargs)
    #     md5 = hashlib.md5(repr(sorted(args.items())).encode('utf-8'))
    #     cache_tag = f'{dataset.name}-{get_feature_detector_name(detector_url)}-{md5.hexdigest()}'
    #     cache_file = dnnlib.make_cache_dir_path('gan-metrics', cache_tag + '.pkl')
    #
    #     # Check if the file exists (all processes must agree).
    #     flag = os.path.isfile(cache_file) if opts.rank == 0 else False
    #     if opts.num_gpus > 1:
    #         flag = torch.as_tensor(flag, dtype=torch.float32, device=opts.device)
    #         torch.distributed.broadcast(tensor=flag, src=0)
    #         flag = (float(flag.cpu()) != 0)
    #
    #     # Load.
    #     if flag:
    #         return FeatureStats.load(cache_file)

    # Initialize.
    num_items = len(dataset)
    if max_items is not None:
        num_items = min(num_items, max_items)
    stats = FeatureStats(max_items=num_items, **stats_kwargs)
    progress = opts.progress.sub(tag='dataset features', num_items=num_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    # Main loop.
    item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
    for images, _labels in torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs):
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector(images.to(opts.device), **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)

    # Save to cache.
    if cache_file is not None and opts.rank == 0:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        temp_file = cache_file + '.' + uuid.uuid4().hex
        stats.save(temp_file)
        os.replace(temp_file, cache_file) # atomic
    return stats


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

    if archi_name == 'StyleGANv2ADA':
        model = exp.get_model()
    elif archi_name == 'StyleGANv3':
        model = exp.get_model(batch_size=1)
        ckpt_state = {
            "start_epoch": 0,
            "model": model.state_dict(),
        }
        torch.save(model.state_dict(), "pytorch_fullyConnectedLayer.pth")
    else:
        raise NotImplementedError("Architectures \'{}\' is not implemented.".format(archi_name))


    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if args.demo == "fid50k_full":
        # 不同的算法输入不同，新增算法时这里也要增加elif
        if archi_name == 'StyleGANv2ADA' or archi_name == 'StyleGANv3':
            # 加载模型权重
            if args.ckpt is None:
                ckpt_file = os.path.join(file_name, "best_ckpt.pth")
            else:
                ckpt_file = args.ckpt
            logger.info("loading checkpoint")
            ckpt = torch.load(ckpt_file, map_location="cpu")
            model = load_ckpt(model, ckpt["model"])
            logger.info("loaded checkpoint done.")

            seeds = args.seeds
            seeds = seeds.split(',')
            seeds = [int(seed) for seed in seeds]
            current_time = time.localtime()

            for seed in seeds:
                z = np.random.RandomState(seed).randn(1, model.z_dim)
                z = torch.from_numpy(z)
                z = z.float()
                if args.device == "gpu":
                    z = z.cuda()
                    if args.fp16:
                        z = z.half()  # to FP16
                data = {
                    'z': z,
                }
                model.setup_input(data)
                with torch.no_grad():
                    img_bgr = model.test_iter()
                    if args.save_result:
                        save_folder = os.path.join(
                            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
                        )
                        os.makedirs(save_folder, exist_ok=True)
                        save_file_name = os.path.join(save_folder, f'seed{seed:08d}.png')
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
            model = load_ckpt(model, ckpt["model"])
            logger.info("loaded checkpoint done.")

            row_seeds = args.row_seeds.split(',')
            row_seeds = [int(seed) for seed in row_seeds]
            col_seeds = args.col_seeds.split(',')
            col_seeds = [int(seed) for seed in col_seeds]
            col_styles = args.col_styles.split(',')
            col_styles = [int(seed) for seed in col_styles]
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
                img_bgr = model.style_mixing(row_seeds, col_seeds, all_seeds, col_styles)
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
