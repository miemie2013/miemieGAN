#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
from loguru import logger

import cv2
import copy
import json
import torch
import numpy as np
from pycocotools.coco import COCO

import PIL
from PIL import Image
try:
    import pyspng
except ImportError:
    pyspng = None

from .. import RandomShapeSingle, YOLOXResizeImage
from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset


class COCODataset(Dataset):
    """
    COCO dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        json_file="instances_train2017.json",
        ann_folder="annotations",
        name="train2017",
        img_size=(416, 416),
        preproc=None,
        cache=False,
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            ann_folder (str): COCO annotations folder name (e.g. 'annotations')
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__(img_size)
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "COCO")
        self.data_dir = data_dir
        self.json_file = json_file
        self.ann_folder = ann_folder

        self.coco = COCO(os.path.join(self.data_dir, self.ann_folder, self.json_file))
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in cats])
        self.imgs = None
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        self.annotations = self._load_coco_annotations()
        if cache:
            self._cache_images()

    def __len__(self):
        return len(self.ids)

    def __del__(self):
        del self.imgs

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def _cache_images(self):
        logger.warning(
            "\n********************************************************************************\n"
            "You are using cached images in RAM to accelerate training.\n"
            "This requires large system RAM.\n"
            "Make sure you have 200G+ RAM and 136G available disk space for training COCO.\n"
            "********************************************************************************\n"
        )
        max_h = self.img_size[0]
        max_w = self.img_size[1]
        cache_file = self.data_dir + "/img_resized_cache_" + self.name + ".array"
        if not os.path.exists(cache_file):
            logger.info(
                "Caching images for the first time. This might take about 20 minutes for COCO"
            )
            self.imgs = np.memmap(
                cache_file,
                shape=(len(self.ids), max_h, max_w, 3),
                dtype=np.uint8,
                mode="w+",
            )
            from tqdm import tqdm
            from multiprocessing.pool import ThreadPool

            NUM_THREADs = min(8, os.cpu_count())
            loaded_images = ThreadPool(NUM_THREADs).imap(
                lambda x: self.load_resized_img(x),
                range(len(self.annotations)),
            )
            pbar = tqdm(enumerate(loaded_images), total=len(self.annotations))
            for k, out in pbar:
                self.imgs[k][: out.shape[0], : out.shape[1], :] = out.copy()
            self.imgs.flush()
            pbar.close()
        else:
            logger.warning(
                "You are using cached imgs! Make sure your dataset is not changed!!\n"
                "Everytime the self.input_size is changed in your exp file, you need to delete\n"
                "the cached data and re-generate them.\n"
            )

        logger.info("Loading cached imgs...")
        self.imgs = np.memmap(
            cache_file,
            shape=(len(self.ids), max_h, max_w, 3),
            dtype=np.uint8,
            mode="r+",
        )

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 5))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        return (res, img_info, resized_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        file_name = self.annotations[index][3]

        img_file = os.path.join(self.data_dir, self.name, file_name)

        img = cv2.imread(img_file)
        assert img is not None

        return img

    def pull_item(self, index):
        id_ = self.ids[index]

        res, img_info, resized_info, _ = self.annotations[index]
        if self.imgs is not None:
            pad_img = self.imgs[index]
            img = pad_img[: resized_info[0], : resized_info[1], :].copy()
        else:
            img = self.load_resized_img(index)

        return img, res.copy(), img_info, np.array([id_])

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w.
                h, w (int): original shape of the image
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, img_id


# 数据清洗
def data_clean(coco, img_ids, catid2clsid, image_dir, type):
    records = []
    ct = 0
    for img_id in img_ids:
        img_anno = coco.loadImgs(img_id)[0]
        im_fname = img_anno['file_name']
        im_w = float(img_anno['width'])
        im_h = float(img_anno['height'])

        ins_anno_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)   # 读取这张图片所有标注anno的id
        instances = coco.loadAnns(ins_anno_ids)   # 这张图片所有标注anno。每个标注有'segmentation'、'bbox'、...

        bboxes = []
        anno_id = []    # 注解id
        for inst in instances:
            x, y, box_w, box_h = inst['bbox']   # 读取物体的包围框
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(im_w - 1, x1 + max(0, box_w - 1))
            y2 = min(im_h - 1, y1 + max(0, box_h - 1))
            if inst['area'] > 0 and x2 >= x1 and y2 >= y1:
                inst['clean_bbox'] = [x1, y1, x2, y2]   # inst增加一个键值对
                bboxes.append(inst)   # 这张图片的这个物体标注保留
                anno_id.append(inst['id'])
            else:
                logger.warn(
                    'Found an invalid bbox in annotations: im_id: {}, '
                    'area: {} x1: {}, y1: {}, x2: {}, y2: {}.'.format(
                        img_id, float(inst['area']), x1, y1, x2, y2))
        num_bbox = len(bboxes)   # 这张图片的物体数

        # 左上角坐标+右下角坐标+类别id
        gt_bbox = np.zeros((num_bbox, 4), dtype=np.float32)
        gt_class = np.zeros((num_bbox, 1), dtype=np.int32)
        gt_score = np.ones((num_bbox, 1), dtype=np.float32)   # 得分的标注都是1
        is_crowd = np.zeros((num_bbox, 1), dtype=np.int32)
        difficult = np.zeros((num_bbox, 1), dtype=np.int32)
        gt_poly = [None] * num_bbox

        for i, box in enumerate(bboxes):
            catid = box['category_id']
            gt_class[i][0] = catid2clsid[catid]
            gt_bbox[i, :] = box['clean_bbox']
            is_crowd[i][0] = box['iscrowd']
            if 'segmentation' in box:
                gt_poly[i] = box['segmentation']

        im_fname = os.path.join(image_dir,
                                im_fname) if image_dir else im_fname
        coco_rec = {
            'im_file': im_fname,
            'im_id': np.array([img_id]),
            'h': im_h,
            'w': im_w,
            'is_crowd': is_crowd,
            'gt_class': gt_class,
            'anno_id': anno_id,
            'gt_bbox': gt_bbox,
            'gt_score': gt_score,
            'gt_poly': gt_poly,
        }

        # logger.debug('Load file: {}, im_id: {}, h: {}, w: {}.'.format(im_fname, img_id, im_h, im_w))
        records.append(coco_rec)   # 注解文件。
        ct += 1
    logger.info('{} samples in {} set.'.format(ct, type))
    return records


def get_class_msg(anno_path):
    _catid2clsid = {}
    _clsid2catid = {}
    _clsid2cname = {}
    with open(anno_path, 'r', encoding='utf-8') as f2:
        dataset_text = ''
        for line in f2:
            line = line.strip()
            dataset_text += line
        eval_dataset = json.loads(dataset_text)
        categories = eval_dataset['categories']
        for clsid, cate_dic in enumerate(categories):
            catid = cate_dic['id']
            cname = cate_dic['name']
            _catid2clsid[catid] = clsid
            _clsid2catid[clsid] = catid
            _clsid2cname[clsid] = cname
    class_names = []
    num_classes = len(_clsid2cname.keys())
    for clsid in range(num_classes):
        class_names.append(_clsid2cname[clsid])
    return _catid2clsid, _clsid2catid, _clsid2cname, class_names


class PPYOLO_COCOEvalDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, json_file, ann_folder, name, cfg, transforms):
        self.data_dir = data_dir
        self.json_file = json_file
        self.ann_folder = ann_folder
        self.name = name

        # 验证集
        val_path = os.path.join(self.data_dir, self.ann_folder, self.json_file)
        val_pre_path = os.path.join(self.data_dir, self.name)

        # 种类id
        _catid2clsid, _clsid2catid, _clsid2cname, class_names = get_class_msg(val_path)

        val_dataset = COCO(val_path)
        val_img_ids = val_dataset.getImgIds()

        keep_img_ids = []  # 只跑有gt的图片，跟随PaddleDetection
        for img_id in val_img_ids:
            ins_anno_ids = val_dataset.getAnnIds(imgIds=img_id, iscrowd=False)  # 读取这张图片所有标注anno的id
            if len(ins_anno_ids) == 0:
                continue
            keep_img_ids.append(img_id)
        val_img_ids = keep_img_ids

        val_records = data_clean(val_dataset, val_img_ids, _catid2clsid, val_pre_path, 'val')

        self.coco = val_dataset
        self.records = val_records
        self.context = cfg.context
        self.transforms = transforms
        self.catid2clsid = _catid2clsid
        self.clsid2catid = _clsid2catid
        self.num_record = len(val_records)
        self.indexes = [i for i in range(self.num_record)]

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        img_idx = self.indexes[idx]
        sample = copy.deepcopy(self.records[img_idx])

        # transforms
        for transform in self.transforms:
            sample = transform(sample, self.context)

        # 取出感兴趣的项
        pimage = sample['image']
        im_size = np.array([sample['h'], sample['w']]).astype(np.float32)
        id = sample['im_id']
        return pimage, im_size, id


class StyleGANv2ADADataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, resolution=None,
                 max_size=None, use_labels=False, xflip=False, random_seed=0, len_phases=4, batch_size=1):

        self.dataroot = dataroot
        self.len_phases = len_phases

        self._type = 'dir'
        self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self.dataroot) for root, _dirs, files in os.walk(self.dataroot) for fname in files}

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self.dataroot))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')

        # 父类
        # super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

        # 一轮的步数。丢弃最后几个样本。
        self.batch_size = batch_size
        self.train_steps = self._raw_idx.size // batch_size

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self.dataroot)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self.dataroot, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

    def __len__(self):
        size = self._raw_idx.size
        return size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        image_gen_c = [self.get_label(np.random.randint(len(self))) for _ in range(self.len_phases)]
        return image.copy(), self.get_label(idx), image_gen_c, self._raw_idx[idx]

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])


    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)


class StyleGANv2ADATestDataset(torch.utils.data.Dataset):
    def __init__(self, seeds, z_dim):
        self.seeds = seeds
        self.z_dim = z_dim

    def __len__(self):
        size = len(self.seeds)
        return size

    def __getitem__(self, idx):
        seed = self.seeds[idx]
        z = np.random.RandomState(seed).randn(self.z_dim, )
        datas = {
            'z': z,
            'seed': seed,
        }
        return datas



from itertools import chain
from pathlib import Path


def listdir(dname):
    # targets = ['png', 'jpg', 'jpeg', 'JPG']
    targets = ['png', 'jpg', 'jpeg']   # 为了去重
    fnames = list(
        chain(*[
            list(Path(dname).rglob('*.' + ext))
            for ext in targets
        ]))
    # 这里是咩酱加上的代码，windows系统下'jpg'和'JPG'后缀的图片重复，所以去重。
    # fnames2 = []
    # for i, fn in enumerate(fnames):
    #     if fn not in fnames2:
    #         fnames2.append(fn)
    # fnames = fnames2
    return fnames


def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))


class ImageFolder(Dataset):
    def __init__(self, root, use_sampler=False):
        self.samples, self.targets = self._make_dataset(root)
        self.use_sampler = use_sampler
        if self.use_sampler:
            self.sampler = _make_balanced_sampler(self.targets)
            self.iter_sampler = iter(self.sampler)

    def _make_dataset(self, root):
        domains = os.listdir(root)
        fnames, labels = [], []
        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)
            fnames += cls_fnames
            labels += [idx] * len(cls_fnames)
        # indexes = [i for i in range(len(fnames))]
        # np.random.shuffle(indexes)
        # fnames2, labels2 = [], []
        # for i in indexes:
        #     fnames2.append(fnames[i])
        #     labels2.append(labels[i])
        return fnames, labels
        # return fnames2, labels2

    def __getitem__(self, i):
        if self.use_sampler:
            try:
                index = next(self.iter_sampler)
            except StopIteration:
                self.iter_sampler = iter(self.sampler)
                index = next(self.iter_sampler)
        else:
            index = i
        fname = self.samples[index]
        label = self.targets[index]
        return fname, label

    def __len__(self):
        return len(self.targets)


class ReferenceDataset(Dataset):
    def __init__(self, root, use_sampler=None):
        self.samples, self.targets = self._make_dataset(root)
        self.use_sampler = use_sampler
        if self.use_sampler:
            self.sampler = _make_balanced_sampler(self.targets)
            self.iter_sampler = iter(self.sampler)

    def _make_dataset(self, root):
        domains = os.listdir(root)
        fnames, fnames2, labels = [], [], []
        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)
            fnames += cls_fnames
            fnames2 += random.sample(cls_fnames, len(cls_fnames))
            labels += [idx] * len(cls_fnames)
        return list(zip(fnames, fnames2)), labels

    def __getitem__(self, i):
        if self.use_sampler:
            try:
                index = next(self.iter_sampler)
            except StopIteration:
                self.iter_sampler = iter(self.sampler)
                index = next(self.iter_sampler)
        else:
            index = i
        fname, fname2 = self.samples[index]
        label = self.targets[index]
        return fname, fname2, label

    def __len__(self):
        return len(self.targets)


class StarGANv2Dataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, is_train, preprocess, test_count=0):
        self.preprocess = preprocess
        self.dataroot = dataroot
        self.is_train = is_train
        if self.is_train:
            self.src_loader = ImageFolder(self.dataroot, use_sampler=True)
            self.ref_loader = ReferenceDataset(self.dataroot, use_sampler=True)
            self.counts = len(self.src_loader)
        else:
            files = os.listdir(self.dataroot)
            if 'src' in files and 'ref' in files:
                self.src_loader = ImageFolder(os.path.join(
                    self.dataroot, 'src'))
                self.ref_loader = ImageFolder(os.path.join(
                    self.dataroot, 'ref'))
            else:
                self.src_loader = ImageFolder(self.dataroot)
                self.ref_loader = ImageFolder(self.dataroot)
            self.counts = min(test_count, len(self.src_loader))
            self.counts = min(self.counts, len(self.ref_loader))

    def __len__(self):
        size = len(self.seeds)
        return size

    def __getitem__(self, idx):
        seed = self.seeds[idx]
        z = np.random.RandomState(seed).randn(self.z_dim, )
        datas = {
            'z': z,
            'seed': seed,
        }
        return datas


