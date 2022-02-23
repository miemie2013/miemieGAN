#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @miemie2013

import importlib
import os
import sys


def get_exp_by_file(exp_file):
    try:
        sys.path.append(os.path.dirname(exp_file))
        current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])
        exp = current_exp.Exp()
    except Exception:
        raise ImportError("{} doesn't contains class named 'Exp'".format(exp_file))
    return exp



def get_exp(exp_file):
    """
    get Exp object by file.

    Args:
        exp_file (str): file path of experiment.
    """
    assert (
        exp_file is not None
    ), "plz provide exp file."
    if exp_file is not None:
        return get_exp_by_file(exp_file)
