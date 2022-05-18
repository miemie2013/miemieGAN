

import torch
import paddle
import os
import numpy as np
import sys

# add python path of this repo to sys.path
import sys
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from mmgan.models.networks import inception_pytorch

torch.backends.cudnn.benchmark = True  # Improves training speed.
torch.backends.cuda.matmul.allow_tf32 = False  # Allow PyTorch to internally use tf32 for matmul
torch.backends.cudnn.allow_tf32 = False  # Allow PyTorch to internally use tf32 for convolutions



model = torch.load('inception-2015-12-05.pt', map_location=torch.device('cpu'))
std1 = model.state_dict()
save_name_pth = 'inception-2015-12-05.pth'
model.eval()


model2 = inception_pytorch.Inception_v3()
std2 = model2.state_dict()
model2.eval()


already_used = []


use_gpu = True

gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
place = paddle.CUDAPlace(gpu_id) if use_gpu else paddle.CPUPlace()


def copy(name, w, std):
    value2 = paddle.to_tensor(w, place=place)
    value = std[name]
    value = value * 0 + value2
    std[name] = value




for key2, value2 in std2.items():
    if '.bn.num_batches_tracked' in key2:
        continue
    if '.bn.weight' in key2:
        continue
    if '.conv.weight' in key2:
        key1 = key2.replace('.conv.weight', '.weight')
        std2[key2] = std1[key1]
    if '.bn.bias' in key2:
        key1 = key2.replace('.bn.bias', '.beta')
        std2[key2] = std1[key1]
    if '.bn.running_mean' in key2:
        key1 = key2.replace('.bn.running_mean', '.mean')
        std2[key2] = std1[key1]
    if '.bn.running_var' in key2:
        key1 = key2.replace('.bn.running_var', '.var')
        std2[key2] = std1[key1]

std2['output.weight'] = std1['output.weight']
std2['output.bias'] = std1['output.bias']

model2.load_state_dict(std2)
torch.save(std2, save_name_pth)

x_shape = [4, 3, 512, 512]
images = torch.randn(x_shape)
images2 = images.cpu().detach().numpy()
images2 = paddle.to_tensor(images2)

return_features = False
return_features = True

use_fp16 = False
# use_fp16 = True

no_output_bias = False
# no_output_bias = True

code = model.code
print(code)

features = model(images, return_features=return_features, use_fp16=use_fp16, no_output_bias=no_output_bias)
features2 = model2(images, return_features=return_features, use_fp16=use_fp16, no_output_bias=no_output_bias)

# features = model.layers(images)
# features2 = model2.layers(images)


ddd = np.sum((features2.cpu().detach().numpy() - features.cpu().detach().numpy()) ** 2)
print('diff=%.6f (%s)' % (ddd, 'features'))


print()



