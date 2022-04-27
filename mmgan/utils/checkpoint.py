#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import os
import shutil
from loguru import logger

import torch


def load_ckpt(model, ckpt):
    model_state_dict = model.state_dict()
    load_dict = {}
    for key_model, v in model_state_dict.items():
        if key_model not in ckpt:
            logger.warning(
                "{} is not in the ckpt. Please double check and see if this is desired.".format(
                    key_model
                )
            )
            continue
        v_ckpt = ckpt[key_model]
        if v.shape != v_ckpt.shape:
            logger.warning(
                "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
                    key_model, v_ckpt.shape, key_model, v.shape
                )
            )
            continue
        load_dict[key_model] = v_ckpt

    model.load_state_dict(load_dict, strict=False)
    return model


def save_checkpoint(state, is_best, save_dir, model_name="", max_keep=10):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, model_name + ".pth")
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save_dir, "best_ckpt.pth")
        shutil.copyfile(filename, best_filename)
    # delete models
    path_dir = os.listdir(save_dir)
    epoch_ids = []
    for name in path_dir:
        sss = name.split('.')
        if sss[-1] == "pth" or sss[-1] == "pt":
            if sss[0].isdigit():
                epoch_id = int(sss[0])
                epoch_ids.append(epoch_id)
            elif '_' in sss[0]:
                ss = sss[0].split('_')
                if ss[0].isdigit() and ss[1].isdigit():
                    epoch_id = int(ss[0])
                    epoch_ids.append(epoch_id)
    if len(epoch_ids) > max_keep * 1:
        target_epoch_id = min(epoch_ids)
        # 寻找最小iter_id
        target_iter_id = None
        for name in path_dir:
            sss = name.split('.')
            if sss[-1] == "pth" or sss[-1] == "pt":
                if sss[0].isdigit():
                    epoch_id = int(sss[0])
                    iter_id = 0
                    if epoch_id == target_epoch_id:
                        if target_iter_id is None:
                            target_iter_id = iter_id
                        else:
                            if iter_id < target_iter_id:
                                target_iter_id = iter_id
                elif '_' in sss[0]:
                    ss = sss[0].split('_')
                    if ss[0].isdigit() and ss[1].isdigit():
                        epoch_id = int(ss[0])
                        iter_id = int(ss[1])
                        if epoch_id == target_epoch_id:
                            if target_iter_id is None:
                                target_iter_id = iter_id
                            else:
                                if iter_id < target_iter_id:
                                    target_iter_id = iter_id
        if target_iter_id == 0:
            del_model = '%s/%d.pth' % (save_dir, target_epoch_id)
        else:
            del_model = '%s/%d_%d.pth' % (save_dir, target_epoch_id, target_iter_id)
        if os.path.exists(del_model):
            os.remove(del_model)
