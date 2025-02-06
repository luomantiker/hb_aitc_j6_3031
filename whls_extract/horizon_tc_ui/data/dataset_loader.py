# Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import os

import cv2
import numpy as np
import skimage.io
from PIL import Image

from horizon_tc_ui.data.loader import Loader


class ImageNetLoader(Loader):
    """
    a data loader for ImageNet valid dataset
    """
    def __init__(self, image_path, label_path=None, imread_mode='skimage'):
        super(ImageNetLoader, self).__init__()
        if imread_mode == 'opencv':
            self.image_read_method = cv2.imread
        elif imread_mode == 'skimage' or imread_mode == 'caffe':
            self.image_read_method = lambda x: skimage.img_as_float(
                skimage.io.imread(x)).astype(np.float32)
        elif imread_mode == 'pil':
            self.image_read_method = lambda x: np.array(
                Image.open(x).convert('RGB'))
        else:
            raise ValueError(
                "Unsupport image read method:{}".format(imread_mode))

        if label_path:
            file_list, img2label = self.build_im2label(image_path, label_path)
            self._gen = self._generator(file_list, img2label)
        else:  # there is no label txt, only return image
            file_list = self.build_image_list(image_path)
            self._gen = self._generator(file_list)

    def build_image_list(self, data_dir):
        image_name_list = []
        image_file_list = sorted(os.listdir(data_dir))
        for image in image_file_list:
            image_name_list.append(os.path.join(data_dir, image))
        return image_name_list

    def build_im2label(self, image_path, label_path):
        img2label = dict()
        image_name_list = []
        with open(label_path) as file:
            line = file.readline()
            while line:
                img, label = line[:-1].split(" ")
                one_image = os.path.join(image_path, img)
                img2label[one_image] = int(label)
                image_name_list.append(one_image)
                line = file.readline()
        return image_name_list, img2label

    def _generator(self, file_list, img2label=None):
        for image_path in file_list:
            image = self.image_read_method(image_path).astype(np.float32)

            if image.ndim != 3:  # expend gray scale image to three channels
                image = image[..., np.newaxis]
                image = np.concatenate([image, image, image], axis=-1)
            if img2label:
                label = img2label[image_path]
                yield [image, label]
            else:
                yield [image]

    def perform(self):
        return next(self._gen)


class SingleImageLoader(Loader):
    """
    A basic image loader, it will imread one single image for model inference
    """
    def __init__(self, image_path, imread_mode='skimage'):
        super(SingleImageLoader, self).__init__()
        if imread_mode == 'opencv':
            self.image_read_method = cv2.imread
        elif imread_mode == 'skimage' or imread_mode == 'caffe':
            self.image_read_method = lambda x: skimage.img_as_float(
                skimage.io.imread(x)).astype(np.float32)
        elif imread_mode == 'pil':
            self.image_read_method = lambda x: np.array(
                Image.open(x).convert('RGB'))
        else:
            raise ValueError(
                "Unsupport image read method:{}".format(imread_mode))
        self.image_path = image_path

    def perform(self):
        return [self.image_read_method(self.image_path)]


class COCOValidLoader(Loader):
    """
    A generator for coco valid dataset
    """
    def __init__(self, imageset_path, annotations_path, imread_mode='opencv'):
        COCO_CLASSES = [  # noqa
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",  # noqa
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]
        from pycocotools.coco import COCO
        super(COCOValidLoader, self).__init__()
        if imread_mode == 'opencv':
            self.image_read_method = cv2.imread
        elif imread_mode == 'skimage' or imread_mode == 'caffe':
            self.image_read_method = lambda x: skimage.img_as_float(
                skimage.io.imread(x)).astype(np.float32)
        else:
            raise ValueError(
                "Unsupport image read method:{}".format(imread_mode))

        if annotations_path:
            self.annotations_path = annotations_path
            self.imageset_path = imageset_path
            self.classes = COCO_CLASSES
            self.coco = COCO(self.annotations_path)
            self.image_ids = sorted(self.coco.getImgIds())
            class_cat = self.coco.dataset["categories"]
            self.id2name = {}
            for (i, cat) in enumerate(class_cat):
                self.id2name[cat['id']] = cat['name']

            self._gen = self._generator()
        else:
            self.imageset_path = imageset_path
            self._gen = self._generator_without_anno()

    def _generator_without_anno(self):
        """calibration data generator without annotation"""
        file_name_dir = sorted(os.listdir(self.imageset_path))
        for file in file_name_dir:
            image_path = os.path.join(self.imageset_path, file)
            image = self.image_read_method(image_path).astype(np.float32)
            yield [image]

    def _generator(self):
        for entry in self.coco.loadImgs(self.image_ids):
            filename = entry['file_name']
            image = self.image_read_method(
                os.path.join(self.imageset_path, filename)).astype(np.float32)
            org_height, org_width, _ = image.shape

            ann_ids = self.coco.getAnnIds(imgIds=entry['id'])
            annotations = self.coco.loadAnns(ann_ids)

            height = entry['height']
            width = entry['width']

            info_dict = {}
            info_dict['origin_shape'] = (org_height, org_width)
            info_dict['image_name'] = filename
            info_dict['class_name'] = []
            info_dict['class_id'] = []
            info_dict['bbox'] = []
            if len(annotations) > 0:
                for ann in annotations:
                    x1, y1, w, h = ann['bbox']
                    x2 = x1 + w
                    y2 = y1 + h
                    x1 = np.minimum(width, np.maximum(0, x1))
                    y1 = np.minimum(height, np.maximum(0, y1))
                    x2 = np.minimum(width, np.maximum(0, x2))
                    y2 = np.minimum(height, np.maximum(0, y2))
                    cat_name = self.id2name[ann['category_id']]
                    class_id = self.classes.index(cat_name)
                    info_dict['class_name'].append(cat_name)
                    info_dict['class_id'].append(class_id)
                    info_dict['bbox'].append([x1, y1, x2, y2])
            yield [image, info_dict]

    def perform(self):
        return next(self._gen)


class VOCValidLoader(Loader):
    """
    A generator for voc valid dataset
    """
    def __init__(self,
                 imageset_path,
                 dataset_path,
                 val_txt_path,
                 imread_mode='opencv',
                 segmentation=False):
        VOC_CLASSES = {
            "aeroplane": (0, "Vehicle"),
            "bicycle": (1, "Vehicle"),
            "bird": (2, "Animal"),
            "boat": (3, "Vehicle"),
            "bottle": (4, "Indoor"),
            "bus": (5, "Vehicle"),
            "car": (6, "Vehicle"),
            "cat": (7, "Animal"),
            "chair": (8, "Indoor"),
            "cow": (9, "Animal"),
            "diningtable": (10, "Indoor"),
            "dog": (11, "Animal"),
            "horse": (12, "Animal"),
            "motorbike": (13, "Vehicle"),
            "person": (14, "Person"),
            "pottedplant": (15, "Indoor"),
            "sheep": (16, "Animal"),
            "sofa": (17, "Indoor"),
            "train": (18, "Vehicle"),
            "tvmonitor": (19, "Indoor"),
        }
        VOC_YEARS = ["VOC2007", "VOC2012"]  # noqa

        super(VOCValidLoader, self).__init__()
        if imread_mode == 'opencv':
            self.image_read_method = cv2.imread
        elif imread_mode == 'skimage' or imread_mode == 'caffe':
            self.image_read_method = lambda x: skimage.img_as_float(
                skimage.io.imread(x)).astype(np.float32)
        else:
            raise ValueError(
                "Unsupport image read method:{}".format(imread_mode))

        self.segmentation = segmentation
        if dataset_path and val_txt_path:
            if self.segmentation is True:
                self.seg_path = os.path.join(dataset_path, "SegmentationClass")
            self.annotations_path = os.path.join(dataset_path, "Annotations")
            self.imageset_path = os.path.join(dataset_path, "JPEGImages")
            self.val_txt_path = val_txt_path
            self.classes = VOC_CLASSES
            self._gen = self._generator()
        elif imageset_path is not None:
            self.imageset_path = imageset_path
            self._gen = self._generator_without_anno()
        else:
            raise ValueError(
                "imageset_path or (dataset_path and val_txt_path) is not set ")

    def _generator_without_anno(self):
        """calibration data generator without annotation"""
        file_name_dir = sorted(os.listdir(self.imageset_path))
        for file in file_name_dir:
            image_path = os.path.join(self.imageset_path, file)
            image = self.image_read_method(image_path).astype(np.float32)
            yield [image]

    def _generator(self):
        import xml.etree.ElementTree as ET
        val_file = open(self.val_txt_path, 'r')
        for f in val_file:
            file_name = f.strip() + '.xml'
            annotation_path = os.path.join(self.annotations_path, file_name)
            tree = ET.ElementTree(file=annotation_path)
            root = tree.getroot()
            object_set = root.findall('object')
            image_path = root.find('filename').text
            image = self.image_read_method(
                os.path.join(self.imageset_path,
                             image_path)).astype(np.float32)
            org_h, org_w, _ = image.shape
            info_dict = {}

            info_dict['origin_shape'] = (org_h, org_w)
            info_dict['image_name'] = image_path

            if self.segmentation is True:
                seg_file = f.strip() + '.png'
                seg_file = os.path.join(self.seg_path, seg_file)
                seg = Image.open(seg_file)
                seg = np.array(seg)
                seg[seg > 20] = 0
                info_dict['seg'] = seg
            else:
                info_dict['class_name'] = []
                info_dict['class_id'] = []
                info_dict['bbox'] = []
                info_dict["difficult"] = []
                for obj in object_set:
                    obj_name = obj.find('name').text
                    bbox = obj.find('bndbox')
                    x1 = int(bbox.find('xmin').text)
                    y1 = int(bbox.find('ymin').text)
                    x2 = int(bbox.find('xmax').text)
                    y2 = int(bbox.find('ymax').text)
                    difficult = int(obj.find("difficult").text)
                    bbox_loc = [x1, y1, x2, y2]

                    info_dict['class_name'].append(obj_name)
                    info_dict['class_id'].append(self.classes[obj_name][0])
                    info_dict['bbox'].append(bbox_loc)
                    info_dict["difficult"].append(difficult)

            yield [image, info_dict]

    def perform(self):
        return next(self._gen)
