import os
import cv2

dataroot = '../data/data42681/afhq/train/dog'
target_res = 256


save_dataroot = '%s_%d'%(dataroot, target_res)
os.makedirs(save_dataroot, exist_ok=True)
names = os.listdir(dataroot)

for name in names:
    path = os.path.join(dataroot, name)
    save_path = os.path.join(save_dataroot, name)
    img = cv2.imread(path)
    img = cv2.resize(img, (target_res, target_res), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(save_path, img)

print('Done.')




