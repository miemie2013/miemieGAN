

import torch
import numpy as np




ckpt_file1 = 'stylegan3_r_32_19.pth'
state_dict1_pytorch = torch.load(ckpt_file1, map_location=torch.device('cpu'))
state_dict1_pytorch = state_dict1_pytorch['model']

# ckpt_file2 = 'stylegan3_r_32_00.pth'
ckpt_file2 = 'StyleGANv3_outputs/styleganv3_r_32_custom/1.pth'
state_dict2_pytorch = torch.load(ckpt_file2, map_location=torch.device('cpu'))
state_dict2_pytorch = state_dict2_pytorch['model']


# ======================== discriminator ========================
print('======================== discriminator ========================')
for key, value1 in state_dict1_pytorch.items():
    if '_ema' in key:
        continue
    v1 = value1.cpu().detach().numpy()
    value2 = state_dict2_pytorch[key]
    v2 = value2.cpu().detach().numpy()
    ddd = np.sum((v1 - v2) ** 2)
    if ddd > 0.000001:
        print('diff=%.6f (%s)' % (ddd, key))


print('==============================================')
print()


for key, value1 in state_dict1_pytorch.items():
    if '_ema' in key:
        continue
    v1 = value1.cpu().detach().numpy()
    value2 = state_dict2_pytorch[key]
    v2 = value2.cpu().detach().numpy()
    ddd = np.sum((v1 - v2) ** 2)
    if ddd <= 0.000001:
        print('diff=%.6f (%s)' % (ddd, key))





