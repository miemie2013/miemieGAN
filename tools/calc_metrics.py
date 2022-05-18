#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @miemie2013

import argparse
import os
import time

import numpy as np
from loguru import logger
import datetime
import cv2

import torch

# add python path of this repo to sys.path
import sys
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from mmgan.data.data_augment import *
from mmgan.exp import get_exp
from mmgan.models import Inception_v3
from mmgan.utils import fuse_model, get_model_info, postprocess, vis, get_classes, vis2, load_ckpt

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("MieMieGAN Demo!")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)

    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    parser.add_argument("-db", "--dataset_batch_size", type=int, default=64, help="dataset batch size")
    parser.add_argument("-b", "--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("-n", "--num_gen", type=int, default=50000, help="num gen")
    parser.add_argument(
        "--inceptionv3_path",
        default="",
        type=str,
        help="inceptionv3_path",
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

    def append_tensor(self, x, num_gpus=1, rank=0):
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        assert 0 <= rank < num_gpus
        self.append(x.cpu().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_tensor(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov


@torch.no_grad()
def calc_stylegan2ada_metric(exp, inceptionv3_model, model, device, dataset_batch_size, batch_size, num_gen, G_kwargs={}):
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
    dataset = StyleGANv2ADADataset(
        dataroot=exp.dataroot,
        batch_size=dataset_batch_size,
        **exp.dataset_train_cfg,
    )
    n_dataset = len(dataset)
    return_features = True

    sampler = torch.utils.data.SequentialSampler(dataset)
    dataloader_kwargs = {
        "num_workers": 0,
        "pin_memory": True,
        "sampler": sampler,
    }
    dataloader_kwargs["batch_size"] = dataset_batch_size
    test_dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    iter_loader = iter(test_dataloader)
    max_eval_steps = len(test_dataloader)

    num_items = len(test_dataloader)
    real_stats_kwargs = dict(capture_mean_cov=True,)
    real_stats = FeatureStats(max_items=n_dataset, **real_stats_kwargs)

    log_interval = 1024
    for i in range(max_eval_steps):
        n_imgs = i * dataset_batch_size
        if n_dataset < log_interval or n_imgs % log_interval == 0:
            logger.info('dataset features: [%d/%d]' % (n_imgs, n_dataset))

        data = next(iter_loader)
        real_image, label, image_gen_c, _ = data
        real_image = real_image.to(torch.float32)  # RGB格式
        real_image = real_image.to(device)
        real_features = inceptionv3_model(real_image, return_features=return_features)
        real_stats.append_tensor(real_features, num_gpus=1, rank=0)
    mu_real, sigma_real = real_stats.get_mean_cov()

    batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0

    fake_stats_kwargs = dict(capture_mean_cov=True,)
    fake_stats = FeatureStats(max_items=num_gen, **fake_stats_kwargs)

    from collections import deque
    time_stat = deque(maxlen=20)
    start_time = time.time()
    end_time = time.time()
    num_imgs = num_gen
    start = time.time()
    i = 0

    # Main loop.
    while not fake_stats.is_full():
        images = []
        for _i in range(batch_size // batch_gen):
            z = torch.randn([batch_gen, exp.z_dim], dtype=torch.float32)
            z = z.to(device)
            c = [dataset.get_label(np.random.randint(len(dataset))) for _i in range(batch_gen)]
            c = torch.from_numpy(np.stack(c))
            c = c.to(device)
            img = model.gen_images(z=z, c=c, **G_kwargs)
            img = (img * 127.5 + 128)
            img = img.clamp(0, 255)
            images.append(img)
        images = torch.cat(images)  # RGB格式
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        fake_features = inceptionv3_model(images, return_features=return_features)
        fake_stats.append_tensor(fake_features, num_gpus=1, rank=0)

        # 估计剩余时间
        start_time = end_time
        end_time = time.time()
        time_stat.append(end_time - start_time)
        time_cost = np.mean(time_stat)
        eta_sec = (num_imgs - i * batch_size) * time_cost
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        n_imgs = i * batch_size
        if num_gen < log_interval or n_imgs % log_interval == 0:
            logger.info('generator features: [%d/%d], eta=%s.' % (n_imgs, num_gen, eta))

        i += 1
    cost = time.time() - start
    logger.info('total time: {0:.6f}s'.format(cost))
    logger.info('Speed: %.6fs per image,  %.1f FPS.' % ((cost / num_imgs), (num_imgs / cost)))
    mu_gen, sigma_gen = fake_stats.get_mean_cov()

    m = np.square(mu_gen - mu_real).sum()
    import scipy.linalg
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    fid = float(fid)
    logger.info('FID: %.6f' % (fid, ))



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
    elif archi_name == 'StyleGANv3':
        model = exp.get_model(batch_size=1)
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

        # build inceptionv3
        inceptionv3_model = Inception_v3()
        param_dict = torch.load(args.inceptionv3_path, map_location="cpu")
        inceptionv3_model.load_state_dict(param_dict)
        inceptionv3_model.eval()
        inceptionv3_model = inceptionv3_model.to(device)

        # calc stylegan2ada metric
        calc_stylegan2ada_metric(exp, inceptionv3_model, model, device, args.dataset_batch_size, args.batch_size, args.num_gen)
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
        args.inceptionv3_path = '../' + args.inceptionv3_path   # 如果是绝对路径，把这一行注释掉
    exp = get_exp(args.exp_file)

    main(exp, args)
