import argparse
import torch
import numpy as np


def make_parser():
    parser = argparse.ArgumentParser("MieMieGAN Demo!")
    parser.add_argument(
        "--cp1",
        default="",
        type=str,
        help="ckpt_file1",
    )
    parser.add_argument(
        "--cp2",
        default="",
        type=str,
        help="ckpt_file2",
    )
    parser.add_argument(
        "--d_value",
        default=0.00001,
        type=float,
        help="d_value",
    )
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()

    ckpt_file1 = args.cp1
    state_dict1_pytorch = torch.load(ckpt_file1, map_location=torch.device('cpu'))
    state_dict1_pytorch = state_dict1_pytorch['model']

    ckpt_file2 = args.cp2
    state_dict2_pytorch = torch.load(ckpt_file2, map_location=torch.device('cpu'))
    state_dict2_pytorch = state_dict2_pytorch['model']

    d_value = args.d_value
    print('======================== diff(weights) > d_value=%.6f ========================' % d_value)
    for key, value1 in state_dict1_pytorch.items():
        if '_ema' in key:
            continue
        if 'augment_pipe.' in key:
            continue
        v1 = value1.cpu().detach().numpy()
        value2 = state_dict2_pytorch[key]
        v2 = value2.cpu().detach().numpy()
        ddd = np.sum((v1 - v2) ** 2)
        if ddd > d_value:
            print('diff=%.6f (%s)' % (ddd, key))

    print()
    print()
    print('======================== diff(weights) <= d_value=%.6f ========================' % d_value)
    for key, value1 in state_dict1_pytorch.items():
        if '_ema' in key:
            continue
        if 'augment_pipe.' in key:
            continue
        v1 = value1.cpu().detach().numpy()
        value2 = state_dict2_pytorch[key]
        v2 = value2.cpu().detach().numpy()
        ddd = np.sum((v1 - v2) ** 2)
        if ddd <= d_value:
            print('diff=%.6f (%s)' % (ddd, key))





