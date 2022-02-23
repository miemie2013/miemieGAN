#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import datetime
import os
import math
import time
import numpy as np
from loguru import logger

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from mmgan.data import DataPrefetcher, StyleGANv2ADADataPrefetcher
from mmgan.data.data_prefetcher import FCOSDataPrefetcher
from mmgan.utils import (
    MeterBuffer,
    ModelEMA,
    all_reduce_norm,
    get_local_rank,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    is_parallel,
    load_ckpt,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize
)


class Trainer:
    def __init__(self, exp, args):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp
        # 算法名字
        self.archi_name = self.exp.archi_name
        self.args = args

        # training related attr
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.device = "cuda:{}".format(self.local_rank)
        self.use_model_ema = exp.ema

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    def train_one_iter(self):
        iter_start_time = time.time()
        if self.archi_name == 'StyleGANv2ADA':
            phase_real_img, phase_real_c, phases_all_gen_c = self.prefetcher.next()
            phase_real_img = phase_real_img.to(self.data_type)
            phase_real_c = phase_real_c.to(self.data_type)
            phases_all_gen_c = [x.to(self.data_type) for x in phases_all_gen_c]
            phase_real_img.requires_grad = False
            phase_real_c.requires_grad = False
            for x in phases_all_gen_c:
                x.requires_grad = False
            data_end_time = time.time()

            data = [phase_real_img, phase_real_c, phases_all_gen_c]
            self.model.setup_input(data)
            with torch.cuda.amp.autocast(enabled=self.amp_training):
                outputs = self.model.train_iter(optimizers=[])
        else:
            raise NotImplementedError("Architectures \'{}\' is not implemented.".format(self.archi_name))


        loss = outputs["total_loss"]

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        # 梯度裁剪
        if self.need_clip:
            for param_group in self.optimizer.param_groups:
                if param_group['need_clip']:
                    torch.nn.utils.clip_grad_norm_(param_group['params'], max_norm=param_group['clip_norm'], norm_type=2)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_model_ema:
            self.ema_model.update(self.model)

        # 修改学习率
        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        if self.archi_name == 'YOLOX':
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
        elif self.archi_name == 'PPYOLO':
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr * param_group['base_lr'] / self.base_lr   # = lr * 参数自己的学习率
        elif self.archi_name == 'FCOS':
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr * param_group['base_lr'] / self.base_lr   # = lr * 参数自己的学习率
        else:
            raise NotImplementedError("Architectures \'{}\' is not implemented.".format(self.archi_name))

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )

    def before_train(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        # model related init
        torch.cuda.set_device(self.local_rank)
        model = self.exp.get_model()
        # logger.info("Model Summary: {}".format(get_model_info(self.archi_name, model, self.exp.test_size)))
        model.to(self.device)

        # 是否进行梯度裁剪
        self.need_clip = False

        if self.archi_name == 'StyleGANv2ADA':
            learning_rate = self.exp.basic_lr_per_img * self.args.batch_size
            beta1 = self.exp.optimizer_cfg['generator']['beta1']
            beta2 = self.exp.optimizer_cfg['generator']['beta2']

            G_reg_interval = self.exp.G_reg_interval
            D_reg_interval = self.exp.D_reg_interval

            for name, reg_interval in [('G', G_reg_interval), ('D', D_reg_interval)]:
                if reg_interval is None:
                    pass
                    # opt = dnnlib.util.construct_class_by_name(params=module.parameters(),
                    #                                           **opt_kwargs)  # subclass of torch.optim.Optimizer
                    # phases += [dnnlib.EasyDict(name=name + 'both', module=module, opt=opt, interval=1)]
                else:  # Lazy regularization.
                    mb_ratio = reg_interval / (reg_interval + 1)
                    new_lr = learning_rate * mb_ratio
                    new_beta1 = beta1 ** mb_ratio
                    new_beta2 = beta2 ** mb_ratio
                if name == 'G':
                    self.base_lr_G = new_lr
                    self.exp.optimizer_cfg['generator']['beta1'] = new_beta1
                    self.exp.optimizer_cfg['generator']['beta2'] = new_beta2
                elif name == 'D':
                    self.base_lr_D = new_lr
                    self.exp.optimizer_cfg['discriminator']['beta1'] = new_beta1
                    self.exp.optimizer_cfg['discriminator']['beta2'] = new_beta2


            # param_groups = []
            # base_wd = self.exp.weight_decay
            # momentum = self.exp.momentum
            # # 是否进行梯度裁剪
            # self.need_clip = hasattr(self.exp, 'clip_grad_by_norm')
            # self.clip_norm = 1000000.0
            # if self.need_clip:
            #     self.clip_norm = getattr(self.exp, 'clip_grad_by_norm')
            # model.add_param_group(param_groups, self.base_lr, base_wd, self.need_clip, self.clip_norm)

            # solver related init
            self.optimizer_G = self.exp.get_optimizer(self.base_lr_G, 'G')
            self.optimizer_D = self.exp.get_optimizer(self.base_lr_D, 'D')

            # value of epoch will be set in `resume_train`
            model = self.resume_train(model)


            self.train_loader = self.exp.get_data_loader(
                batch_size=self.args.batch_size,
                is_distributed=self.is_distributed,
                cache_img=self.args.cache,
            )

            logger.info("init prefetcher, this might take one minute or less...")
            self.prefetcher = StyleGANv2ADADataPrefetcher(self.train_loader)
        else:
            raise NotImplementedError("Architectures \'{}\' is not implemented.".format(self.archi_name))

        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )
        if self.args.occupy:
            occupy_mem(self.local_rank)

        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, self.exp.ema_decay)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model
        self.model.train()

        self.evaluator = self.exp.get_evaluator(
            batch_size=self.args.eval_batch_size, is_distributed=self.is_distributed
        )

        # Tensorboard logger
        if self.rank == 0:
            self.tblogger = SummaryWriter(self.file_name)

        logger.info("Training start...")
        # logger.info("\n{}".format(model))
        trainable_params = 0
        nontrainable_params = 0
        for name_, param_ in model.named_parameters():
            mul = np.prod(param_.shape)
            if param_.requires_grad is True:
                trainable_params += mul
            else:
                nontrainable_params += mul
        total_params = trainable_params + nontrainable_params
        logger.info('Total params: %s' % format(total_params, ","))
        logger.info('Trainable params: %s' % format(trainable_params, ","))
        logger.info('Non-trainable params: %s' % format(nontrainable_params, ","))

    def after_train(self):
        logger.info(
            "Training of experiment is done."
        )

    def before_epoch(self):
        logger.info("---> start train epoch{}".format(self.epoch + 1))
        if self.archi_name == 'StyleGANv2ADA':
            self.train_loader.dataset.set_epoch(self.epoch)
        else:
            raise NotImplementedError("Architectures \'{}\' is not implemented.".format(self.archi_name))

    def after_epoch(self):
        self.save_ckpt(ckpt_name="%d" % (self.epoch + 1))

        if (self.epoch + 1) % self.exp.eval_interval == 0:
            all_reduce_norm(self.model)
            self.evaluate_and_save_model()

    def before_iter(self):
        pass

    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if (self.iter + 1) % self.exp.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            log_msg = "{}, mem: {:.0f}Mb, {}, {}, lr: {:.6f}".format(progress_str, gpu_mem_usage(), time_str, loss_str, self.meter["lr"].latest, )
            if self.archi_name == 'YOLOX':
                log_msg += (", size: {:d}, {}".format(self.input_size[0], eta_str))
            elif self.archi_name == 'PPYOLO':
                log_msg += (", {}".format(eta_str))
            elif self.archi_name == 'FCOS':
                log_msg += (", {}".format(eta_str))
            else:
                raise NotImplementedError("Architectures \'{}\' is not implemented.".format(self.archi_name))
            logger.info(log_msg)
            self.meter.clear_meters()

        if self.archi_name == 'YOLOX':
            # random resizing
            if (self.progress_in_iter + 1) % 10 == 0:
                self.input_size = self.exp.random_resize(
                    self.train_loader, self.epoch, self.rank, self.is_distributed
                )
        elif self.archi_name == 'PPYOLO':
            pass
        elif self.archi_name == 'FCOS':
            pass
        else:
            raise NotImplementedError("Architectures \'{}\' is not implemented.".format(self.archi_name))

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
            else:
                ckpt_file = self.args.ckpt

            ckpt = torch.load(ckpt_file, map_location=self.device)
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            # resume the training states variables
            start_epoch = ckpt["start_epoch"]
            self.start_epoch = start_epoch
            logger.info(
                "loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, self.start_epoch
                )
            )  # noqa
        else:
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                model = load_ckpt(model, ckpt)
            self.start_epoch = 0

        return model

    def evaluate_and_save_model(self):
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module

        ap50_95, ap50, summary = self.exp.eval(
            evalmodel, self.evaluator, self.is_distributed
        )
        self.model.train()
        if self.rank == 0:
            self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
            self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
            logger.info("\n" + summary)
        synchronize()

        self.save_ckpt("last_epoch", ap50_95 > self.best_ap)
        self.best_ap = max(self.best_ap, ap50_95)

    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )
