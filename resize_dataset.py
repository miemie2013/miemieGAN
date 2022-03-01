import os
import cv2
import numpy as np

# dataroot = '../data/data42681/afhq/train/dog'
# target_res = 256

dataroot = '../data/flowers'
target_res = 512


save_dataroot = '%s_%d'%(dataroot, target_res)
os.makedirs(save_dataroot, exist_ok=True)
names = os.listdir(dataroot)

for name in names:
    path = os.path.join(dataroot, name)
    save_path = os.path.join(save_dataroot, name)
    img = cv2.imread(path)
    h, w, _ = img.shape
    if h == w:
        img = cv2.resize(img, (target_res, target_res), interpolation=cv2.INTER_LINEAR)
    else:
        im_size_min = np.min([h, w])
        im_size_max = np.max([h, w])
        selected_size = target_res
        max_size = target_res
        im_scale = float(selected_size) / float(im_size_min)
        im_scale_x = im_scale
        im_scale_y = im_scale

        img = cv2.resize(
            img,
            None,
            None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=cv2.INTER_LINEAR)
        h, w, _ = img.shape
        if w > h:
            pad = (w - target_res) // 2
            img = img[:, pad:pad+target_res, :]
        else:
            pad = (h - target_res) // 2
            img = img[pad:pad+target_res, :, :]
    cv2.imwrite(save_path, img)

print('Done.')




