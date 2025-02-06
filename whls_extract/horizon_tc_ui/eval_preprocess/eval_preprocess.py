# Copyright (c) 2021 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import inspect
import logging
import os
from typing import Callable, Tuple

import cv2
import numpy as np

from horizon_tc_ui.eval_preprocess import data_transformer
from horizon_tc_ui.eval_preprocess.conf import MODEL_DICT
from horizon_tc_ui.eval_preprocess.dataloader import SingleImageDataLoader
from horizon_tc_ui.version import __version__

__VERSION__ = __version__


class EvalPreprocess:
    def __init__(self,
                 image_dir: str,
                 model_name: str,
                 output_dir: str,
                 single_mode=False,
                 val_txt_path=None) -> None:
        self._check_numpy()
        self.image_dir = image_dir
        self.val_txt_path = val_txt_path
        self.model_name = model_name
        self.output_dir = output_dir[:-1] if output_dir.endswith(
            r'/') else output_dir
        self.single_mode = single_mode
        self.imread_mode = MODEL_DICT[model_name]['read_mode']

        if (not os.path.exists(self.output_dir)) and (not self.single_mode):
            logging.info(f"output dir {self.output_dir} not exist, created!")
            os.mkdir(self.output_dir)
        self.transformers, self.dst = self.get_data_transformer_info()

    def _check_numpy(self) -> None:
        np_version = np.__version__
        if np_version != '1.23.0':
            logging.warning('make sure your numpy version is "1.23.0"')

    def get_data_transformer_info(self) -> Tuple[Callable, Tuple[int, int]]:
        functions_list = [
            name for name, obj in inspect.getmembers(data_transformer)  # noqa
            if inspect.isfunction(obj)
            and inspect.getmodule(obj) == data_transformer
        ]  # noqa
        trans_func_name = f"{self.model_name}_data_transformer"
        if trans_func_name not in functions_list:
            support_list = ', '.join(
                [k for k, v in MODEL_DICT.items() if v['enable'] is True])
            raise ValueError("model_name input wrong,"
                             f"support list: {support_list}")
        return getattr(data_transformer, trans_func_name)()

    def get_image_list(self, image_dir, val_txt) -> list:
        if val_txt:
            with open(val_txt, 'r') as val_file:
                val_dict = {
                    f.strip() + '.jpg': index
                    for index, f in enumerate(val_file)
                }
        image_list = []
        if not os.path.exists(image_dir):
            raise ValueError(f"image dir {image_dir} not exist!!!")

        if os.path.isfile(image_dir):
            logging.info("input single image")
            image_list.append(image_dir)
            return image_list

        for parent, dirnames, filenames in os.walk(image_dir):
            del dirnames
            for filename in filenames:
                if filename.split(".")[-1].lower() in ['jpg', 'jpeg', 'png']:
                    file_path = os.path.join(parent, filename)
                    if val_txt:
                        if val_dict.get(filename) is not None:
                            image_list.append(os.path.abspath(file_path))
                    else:
                        image_list.append(os.path.abspath(file_path))
                else:
                    logging.warning(f"file: {filename} skipped!!")
        if len(image_list) == 0:
            raise ValueError(f"no image can convert in folder: {image_dir}!!!")
        logging.info(f'image list parse completed, {len(image_list)} in total')
        return image_list

    def dump_image(self):
        for image_file in self.get_image_list(self.image_dir,
                                              val_txt=self.val_txt_path):
            image_name = os.path.basename(image_file)
            image_parsed = cv2.imread(image_file)
            if hasattr(image_parsed, 'data'):
                org_h, org_w = image_parsed.shape[:-1]
            else:
                raise ValueError(
                    f'image: {image_name} is not a supported image')
            image = SingleImageDataLoader(self.transformers,
                                          image_file,
                                          imread_mode=self.imread_mode)[0]
            dst_h, dst_w = self.dst
            yield org_h, org_w, dst_h, dst_w, image, image_name

    def save_images(self, image_data):
        for org_h, org_w, dst_h, dst_w, image, image_name in image_data:
            image_affect_file = f'{self.output_dir}/{image_name}_{org_h}_{org_w}_{dst_h}_{dst_w}.bin'  # noqa
            image.tofile(image_affect_file)
            logging.info(f'generated: {image_affect_file}')

    def run(self) -> None:
        if self.single_mode:
            return self.dump_image()
        else:
            self.save_images(image_data=self.dump_image())
