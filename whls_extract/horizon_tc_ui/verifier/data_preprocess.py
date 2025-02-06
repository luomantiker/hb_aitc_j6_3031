# Copyright (c) 2022 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import json
import re
from typing import List, Tuple, Union

import numpy as np

from horizon_tc_ui.config.mapper_consts import preprocess_mode_map
# yapf: disable
from horizon_tc_ui.data.transformer import (BGR2NV12Transformer,
                                            BGR2RGBTransformer,
                                            BGR2YUV444Transformer,
                                            MeanTransformer,
                                            NormalizeTransformer,
                                            RGB2BGRTransformer,
                                            RGB2YUV444Transformer,
                                            ScaleTransformer,
                                            TransposeTransformer,
                                            YUV4442BGRTransformer)
# yapf: enable
from horizon_tc_ui.utils.tool_utils import get_list_from_txt
from horizon_tc_ui.verifier import ModelInfo
from horizon_tc_ui.verifier.params_check import InputInfo, VerifierParams


class VerifierDataPreprocess:
    def __init__(self, pramams_info: VerifierParams) -> None:
        self.params_info = pramams_info
        self.color_convert_map = {
            "yuv444_to_nv12": [YUV4442BGRTransformer, BGR2NV12Transformer],
            "rgb_to_nv12": [RGB2BGRTransformer, BGR2NV12Transformer],
            "rgb_to_yuv444": [RGB2YUV444Transformer],
            "rgb_to_bgr": [RGB2BGRTransformer],
            "bgr_to_nv12": [BGR2NV12Transformer],
            "bgr_to_yuv444": [BGR2YUV444Transformer],
            "bgr_to_rgb": [BGR2RGBTransformer]
        }
        # key from onnx.TensorProto
        self.onnx_data_type_map = {
            12: np.uint32,
            13: np.uint64,
            6: np.int32,
            7: np.int64,
            4: np.float16,
            1: np.float32,
            11: np.float64
        }

    def run(self) -> None:
        if not self.params_info.inputs_info:
            self.generate_data_from_random()
        for idx, model_info in enumerate(self.params_info.models_info):
            if model_info.model_type == "onnx":
                self.preprocess_for_onnx(model_idx=idx)
            elif model_info.model_type in ["bc", "hbm"]:
                if model_info.model_type == "hbm":
                    self.preprocess_for_unchanged(model_idx=idx)
                elif model_info.model_type == "bc" and (
                        model_info.sess.current_phase == "export"
                        or not model_info.desc):
                    self.preprocess_for_unchanged(model_idx=idx)
                else:
                    self.preprocess_for_batch(model_idx=idx)
                    self.preprocess_for_quantized_bc(model_idx=idx)
            else:
                raise ValueError(f'Invalid model type {model_info.model_type} '
                                 f'with {model_info.path}')
        return None

    def generate_data_from_random(self) -> None:
        model_info = self.params_info.models_info[0]
        preprocessed = False
        inputs_info = []
        if model_info.model_type == "bc":
            for graph in model_info.sess.hbir_model.graphs:
                for operation in graph.operations:
                    if (operation.op.name == "hbir.image_convert"
                            or operation.op.name == "hbir.image_preprocess"):
                        preprocessed = True
        elif model_info.model_type == "hbm":
            desc = {k.lower(): v for k, v in model_info.desc.items()}
            input_names = get_list_from_txt(desc['input_names'])
            input_type_rts = get_list_from_txt(desc['input_type_rt'])
            input_type_trains = get_list_from_txt(desc['input_type_train'])
            input_space_and_ranges = get_list_from_txt(
                desc['input_space_and_range'])
            for idx, input_name in enumerate(input_names):
                _input_source = desc['input_source'][input_name]
                _input_type_rt = input_type_rts[idx]
                _input_type_train = input_type_trains[idx]
                _input_space_and_range = input_space_and_ranges[idx]
                if _input_source == "pyramid":
                    preprocessed = True
                    break
                if _input_type_rt == 'nv12' and _input_space_and_range:
                    if _input_space_and_range == "regular":
                        _input_type_rt += '_full'
                    else:
                        _input_type_rt += '_video'
                convert_type = _input_type_rt + '_' + _input_type_train
                if preprocess_mode_map.get(convert_type, 'skip') != 'skip':
                    preprocessed = True
                    break

        for idx, input_name in enumerate(model_info.sess.input_names):
            input_shape = model_info.sess.input_shapes[idx]
            data_type = model_info.sess.input_types[idx]
            if model_info.model_type == "onnx":
                data_type = self.onnx_data_type_map[data_type]

            input_data = self.random_tensor(input_shape, data_type)
            input_info = InputInfo(path="",
                                   name=input_name,
                                   data=input_data,
                                   batch=1,
                                   preprocessed=preprocessed)
            inputs_info.append(input_info)
        self.params_info.inputs_info.append(inputs_info)

    def random_tensor(self, shape: Union[list, tuple],
                      dtype: np.dtype) -> np.ndarray:
        if dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
            return np.random.randint(0,
                                     np.iinfo(dtype).max,
                                     shape,
                                     dtype=dtype)
        elif dtype in [np.int8, np.int16, np.int32, np.int64]:
            return np.random.randint(np.iinfo(dtype).min,
                                     np.iinfo(dtype).max,
                                     shape,
                                     dtype=dtype)
        elif dtype in [np.float16, np.float32, np.float64]:
            return np.random.rand(*shape).astype(dtype)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    def preprocess_for_onnx(self, model_idx: int) -> None:
        """Generate input data for onnx model
        """
        model_info = self.params_info.models_info[model_idx]
        for input_info in self.params_info.inputs_info[model_idx]:
            input_idx = model_info.sess.input_names.index(input_info.name)
            input_shape = tuple(model_info.sess.input_shapes[input_idx])

            input_data = input_info.data
            # expand batch dim
            if input_shape != input_data.shape and input_shape[
                    1:] == input_data.shape:  # noqa
                input_data = np.expand_dims(input_data, axis=0)

            if input_shape[0] != input_info.data.shape[0] and \
                    input_data.shape[0] != 1:
                raise ValueError(
                    "Input data shape does not match model input shape"
                )  # noqa

            # copy data to batch
            if input_shape[0] != input_data.shape[0]:
                input_data = np.tile(input_data, (input_shape[0], ) + (1, ) *
                                     (input_data.ndim - 1))  # noqa
            model_info.inputs.update({input_info.name: input_data})

    def reformat_input_name_with_batch(self, model_info, origin_name):
        desc = {k.lower(): v for k, v in model_info.desc.items()}
        input_name = origin_name
        separate_batch = desc['separate_batch'] == 'True'
        separate_name = desc.get('separate_name', [])
        if input_name.rsplit('_', 1)[0] in separate_name or separate_batch:
            input_name = re.sub(r'_\d+$', '', origin_name)

        if input_name not in desc.get('input_source', {}) and \
                input_name not in model_info.sess.input_names:
            raise ValueError(f"Input name {input_name} not found in model.")
        return input_name

    def preprocess_input_name(self, model_info: ModelInfo,
                              origin_name: str) -> Tuple[str, str]:
        desc = {k.lower(): v for k, v in model_info.desc.items()}
        input_name = origin_name
        suffixes = tuple(['_y', '_uv', '_roi'])
        mode = None  # input.1_1_y/uv/roi -> input.1_1
        if origin_name.endswith(
                suffixes) and origin_name not in desc['input_source']:
            mode = origin_name.rpartition('_')[2]
            input_name = origin_name.replace('_y', '').replace('_uv',
                                                               '').replace(
                                                                   '_roi', '')
        return input_name, mode

    def input_shape_from_desc(self, desc: dict, idx: int) -> List[int]:
        input_shape = desc['input_shape'].split(";")[idx].split("x")
        input_shape = [int(i) for i in input_shape]
        return input_shape

    def preprocess_for_resizer(self, model_info: ModelInfo, input_name: str,
                               input_data: np.ndarray, suffix_name) -> None:
        desc = {k.lower(): v for k, v in model_info.desc.items()}
        idx = desc['input_names'].split(';').index(input_name)
        input_shape = self.input_shape_from_desc(desc, idx)
        separate_name = desc.get('separate_name', [])

        image = input_data.flatten()
        if desc['input_layout_train'].split(";")[idx] == "NCHW":
            input_h, input_w = input_shape[2], input_shape[3]
            input_shape = [
                input_shape[0], input_shape[2], input_shape[3], input_shape[1]
            ]
        else:
            input_h, input_w = input_shape[1], input_shape[2]
        if input_name in separate_name:
            input_shape[0] = 1
        y_shape = input_shape[:-1] + [1]
        uv_shape = input_shape[:1] + [
            int(input_shape[1] / 2),
            int(input_shape[2] / 2), 2
        ]  # noqa
        if desc['input_type_rt'].split(";")[idx] == "gray":
            y_data = image.reshape(y_shape)
            model_info.inputs.update({suffix_name + "_y": y_data})
        else:
            y_data = image[:int(image.size // 1.5)].reshape(y_shape)
            uv_data = image[int(image.size // 1.5):].reshape(uv_shape)
            model_info.inputs.update({suffix_name + "_y": y_data})
            model_info.inputs.update({suffix_name + "_uv": uv_data})
        roi_data = np.array([[0, 0, input_w, input_h]]).astype(np.uint32)
        model_info.inputs.update({suffix_name + "_roi": roi_data})

    def preprocess_for_pyramid(self, model_info: ModelInfo, input_name: str,
                               input_data: np.ndarray, suffix_name) -> None:
        desc = {k.lower(): v for k, v in model_info.desc.items()}
        idx = desc['input_names'].split(';').index(input_name)
        input_type_rt = desc['input_type_rt'].split(';')[idx]
        input_shape = self.input_shape_from_desc(desc, idx)
        seprate_name = desc.get('separate_name', [])

        image = input_data.flatten()
        if desc['input_layout_train'].split(";")[idx] == "NCHW":
            input_shape = [
                input_shape[0], input_shape[2], input_shape[3], input_shape[1]
            ]  # noqa
        if input_name in seprate_name:
            input_shape[0] = 1
        y_shape = input_shape[:-1] + [1]
        uv_shape = input_shape[:1] + [
            int(input_shape[1] / 2),
            int(input_shape[2] / 2), 2
        ]  # noqa
        if input_type_rt == "gray":
            y_data = image.reshape(y_shape)
            model_info.inputs.update({suffix_name + "_y": y_data})
        else:
            y_data = image[:int(image.size // 1.5)].reshape(y_shape)
            uv_data = image[int(image.size // 1.5):].reshape(uv_shape)
            model_info.inputs.update({suffix_name + "_y": y_data})
            model_info.inputs.update({suffix_name + "_uv": uv_data})

    def preprocess_for_unchanged(self, model_idx: int) -> None:
        """Generate input data for unquantized model
        """
        model_info = self.params_info.models_info[model_idx]
        for input_info in self.params_info.inputs_info[model_idx]:
            model_info.inputs.update({input_info.name: input_info.data})

    def preprocess_for_quantized_bc(self, model_idx: int) -> None:
        model_info = self.params_info.models_info[model_idx]
        original_inputs_info = self.params_info.inputs_info[model_idx]
        desc = {k.lower(): v for k, v in model_info.desc.items()}
        for origin_name in model_info.sess.input_names:

            input_name, suffix = self.preprocess_input_name(
                model_info, origin_name)

            name_without_batch = self.reformat_input_name_with_batch(
                model_info, input_name)
            input_info = next((
                info for info in original_inputs_info
                if info.name in [origin_name, input_name, name_without_batch]),
                              None)  # noqa

            if input_info.preprocessed:
                model_info.inputs.update({input_info.name: input_info.data})
                continue

            input_idx = desc['input_names'].split(';').index(
                name_without_batch)  # noqa

            if suffix in ['roi', 'uv']:
                continue

            input_data = self.inverse_preprocess_node_for_quantized_model(
                model_info, input_idx, input_name)

            if desc['input_source'][name_without_batch] == 'resizer':
                self.preprocess_for_resizer(model_info, name_without_batch,
                                            input_data, input_name)  # noqa
            elif desc['input_source'][name_without_batch] == 'pyramid':
                self.preprocess_for_pyramid(model_info, name_without_batch,
                                            input_data, input_name)  # noqa
            else:
                model_info.inputs.update({origin_name: input_data})

    def preprocess_for_batch(self, model_idx: int):
        model_info = self.params_info.models_info[model_idx]
        desc = {k.lower(): v for k, v in model_info.desc.items()}
        if not desc:
            return None
        inputs_info = self.params_info.inputs_info[model_idx]
        input_batch = int(
            desc['input_batch']) if desc['input_batch'] else -1  # noqa
        input_data_process = {}
        if desc['separate_batch'] and desc['separate_batch'] == 'True':
            # separate_batch without input_batch, copy data for every batch
            if input_batch == -1:
                raise ValueError("Wrong input_batch of model.")
            for input_info in inputs_info:
                input_name = input_info.name
                input_data = input_info.data
                for batch_idx in range(input_batch):
                    name = input_name + "_" + str(batch_idx)
                    input_data_process.update({name: input_data})
        elif 'separate_name' in desc.keys() and desc['separate_name'] != '':
            # spearate_batch with separate_name,
            # copy data[batch_idx] for every batch
            separate_names = desc['separate_name'].split(';')
            for input_info in inputs_info:
                input_name = input_info.name
                input_data = input_info.data
                # no separate_name
                if input_name not in separate_names:
                    input_data_process.update({input_name: input_data})
                    continue
                # separate_name without input_shape[0]
                if input_batch == -1:
                    shape_idx = desc['input_names'].split(';').index(
                        input_name)  # noqa
                    separate_input_batch = self.input_shape_from_desc(
                        desc, shape_idx)[0]  # noqa
                    for batch_idx in range(separate_input_batch):
                        name = input_name + "_" + str(batch_idx)
                        input_data_process.update({
                            name: np.expand_dims(input_data[batch_idx], axis=0)
                        })  # noqa
                # separate_name with input_batch
                else:
                    for batch_idx in range(input_batch):
                        name = input_name + "_" + str(batch_idx)
                        input_data_process.update({name: input_data})
        else:
            # no separate_batch and no separate_name but with input_batch
            input_batch = 1 if input_batch == -1 else input_batch
            for input_info in inputs_info:
                input_name = input_info.name
                input_data = input_info.data
                # skip expand dim when input_data has multi batch dim
                if input_data.shape[0] != input_batch:
                    input_data = np.tile(input_data, (input_batch, ) + (1, ) *
                                         (input_data.ndim - 1))
                input_data_process.update({input_name: input_data})
        self.params_info.models_info[model_idx].inputs = input_data_process
        return None

    def inverse_preprocess_node_for_quantized_model(
            self, model_info: ModelInfo, input_idx: int,
            input_name: str) -> np.ndarray:
        """Generate input data for bc model
        """
        desc = {k.lower(): v for k, v in model_info.desc.items()}
        norm_type = desc['norm_type'].split(";")[input_idx]
        input_type_train = desc['input_type_train'].split(";")[input_idx]
        input_type_rt = desc['input_type_rt'].split(";")[input_idx]
        layout = desc['input_layout_train'].split(';')[input_idx][1:]
        color_converted = input_type_train != input_type_rt
        normalized = norm_type != "no_preprocess"
        transposed = (layout == "CHW" and input_type_rt != "featuremap")

        if input_name not in model_info.inputs.keys():
            raise ValueError(f"Input name {input_name} not found in model.")
        input_data = np.copy(model_info.inputs[input_name])
        if input_data.ndim == 3 and normalized:
            input_data = np.expand_dims(input_data, axis=0)
        # NCHW -> NHWC
        if transposed:
            transformed_data = []
            for data in input_data:
                transformed_data.append(
                    TransposeTransformer((1, 2, 0)).run_transform(data))
            input_data = np.array(transformed_data)
            layout = "HWC"

        if normalized:
            if 'scale' in norm_type:
                scale = json.loads(desc['scale_value'].split(';')[input_idx])
                scale = [1 / s for s in scale]
                input_data = ScaleTransformer(np.array(scale),
                                              layout).run_transform(
                                                  input_data)  # noqa
            elif 'std' in norm_type:
                std = json.loads(desc['std_value'].split(';')[input_idx])
                input_data = NormalizeTransformer(
                    np.array(std), layout).run_transform(input_data)
            if 'mean' in norm_type:
                mean = json.loads(desc['mean_value'].split(';')[input_idx])
                mean = [-m for m in mean]
                input_data = MeanTransformer(
                    np.array(mean), layout).run_transform(input_data)  # noqa

        transformers = []
        if color_converted:
            convert_key = input_type_train + "_to_" + input_type_rt  # noqa
            transformers += (self.color_convert_map)[convert_key]

        for transformer in transformers:
            trans = transformer(data_format=layout)
            input_data = trans.run_transform(input_data[0])
            input_data = np.expand_dims(input_data, axis=0)

        if (color_converted or normalized) and input_type_rt != "nv12":
            input_data = (input_data - 128).astype(np.int8)
        return input_data
