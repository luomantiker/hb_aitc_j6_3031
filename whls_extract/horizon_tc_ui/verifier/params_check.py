# Copyright (c) 2022 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import logging
import os
import re
import socket
from copy import deepcopy
from typing import List, Literal

import numpy as np
import onnx
from horizon_nn.api import infer_shapes
from onnx import ValueInfoProto

from horizon_tc_ui import HB_HBIRRuntime, HB_ONNXRuntime
from horizon_tc_ui.hb_hbmruntime import HB_HBMRuntime
from horizon_tc_ui.verifier import InputInfo, ModelInfo, VerifierParams


class VerifierParamsCheck:
    def __init__(self, models_path: str, input_files: list, skip_sim: bool,
                 skip_arm: bool, digits: int, board_ip: str, username: str,
                 password: str) -> None:
        self.models_path = models_path
        self.board_ip = board_ip
        self.input_files = input_files
        self.digits = digits
        self.username = username
        self.password = password
        self.skip_arm = skip_arm
        self.skip_sim = skip_sim
        self.params = VerifierParams()
        self.model_types: List[str] = []
        self.input_repeat = False

    def check(self) -> VerifierParams:
        logging.info(format("check params start"))
        self.check_mode()
        self.check_board_info()
        self.check_digits()
        self.check_model()
        self.check_input()
        self.check_onnx_graph()
        logging.info(format("check params end"))
        return self.params

    def check_model_path(self) -> None:
        models_path = self.models_path.split(',')
        if len(models_path) != 2:
            raise ValueError('Only two models are supported for validation')

        if models_path[0].endswith('bc') and models_path[1].endswith('onnx'):
            models_path[0], models_path[1] = models_path[1], models_path[0]

        for model_path in models_path:
            if not os.path.isfile(model_path):
                raise ValueError("model does not exist: " + model_path)

            if not model_path.endswith(('.onnx', '.bc', '.hbm')):
                raise ValueError(
                    'The model suffix is incorrect. '
                    'The model file supports the following suffix:'
                    'onnx, bc, hbm')

            output_names = []
            model_name, _ = os.path.splitext(os.path.basename(model_path))
            if model_path.endswith('onnx'):
                model = onnx.load(model_path)
                output_names = [output.name for output in model.graph.output]
                model = infer_shapes(model)
                sess = HB_ONNXRuntime(onnx_model=model)
                desc = sess.desc
                model_type = "onnx"
            elif model_path.endswith('.bc'):
                sess = HB_HBIRRuntime(model_file=model_path)
                output_names = sess.output_names
                if self.params.mode == "cosine":
                    sess.register_dump_nodes()
                desc = sess.desc
                model_type = "bc"
            elif model_path.endswith('.hbm'):
                sess = HB_HBMRuntime(model_file=model_path)
                output_names = sess.output_names
                desc = sess.desc
                model_type = "hbm"
            model_dict = ModelInfo(path=os.path.realpath(model_path),
                                   name=model_name,
                                   model_type=model_type,
                                   sess=sess,
                                   desc=desc,
                                   output_names=output_names)
            self.params.models_info.append(model_dict)
            # Reorder to avoid issues caused by the insertion of nodes by
            # layer_out_dump when alternately retrieving output information
            # in subsequent steps.
            self.params.models_info.sort(
                key=lambda model_info: len(model_info.sess.output_names))

        return None

    def check_model(self) -> None:
        self.check_model_path()
        self.check_model_name()
        self.check_model_shape_and_type()
        self.add_onnx_tenser_output()

    def check_mode(self) -> None:
        model_paths = self.models_path.split(",")
        self.model_types = sorted([
            os.path.splitext(os.path.basename(i))[-1][1:] for i in model_paths
        ])
        if self.model_types == ["bc", "hbm"]:
            mode = "consistency"
        elif self.model_types == ["bc", "onnx"]:
            mode = "cosine"
        elif self.model_types == ["bc", "bc"]:
            mode = "cosine"
        elif self.model_types == ["onnx", "onnx"]:
            mode = "cosine"
        else:
            raise ValueError(
                f"{' vs '.join(self.model_types)} is not supported now")
        self.params.mode = mode

    def check_board_info(self) -> None:
        if not self.board_ip:
            return None
        pattern = re.compile(
            r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'  # noqa
        )
        if not re.compile(pattern).match(self.board_ip):
            raise ValueError("board ip invalid: " + self.board_ip)
        logging.info("IP address connectivity is being detected...")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        status = s.connect_ex((self.board_ip, 22))
        if status != 0:
            raise ValueError(f"{self.board_ip} connect failed.")

        logging.info(f"{self.board_ip} connect succeeded.")
        self.params.board_info.ip = self.board_ip
        self.params.board_info.username = self.username
        self.params.board_info.password = self.password
        return None

    def check_digits(self) -> None:
        if self.digits < 0 or self.digits > 16:
            raise ValueError("The digits parameter is in the range [0, 16]."
                             f"You gave the value is {self.digits}.")
        self.params.digits = self.digits
        return None

    def check_version(self) -> None:
        # TODO(ruxin.song): check model version
        # Model and tool versions do not match,
        # please keep the versions consistent
        return None

    def check_input_file(self, model_idx: int, model_info: ModelInfo) -> None:
        input_file_idx = 0
        error_message = []
        preprocessed = True
        input_files_batch = 1
        inputs_info = []
        input_file_contain_names = []

        if self.model_types == ["bc", "hbm"]:
            # only runtime data
            input_names = model_info.sess.input_names
        elif self.model_types == ["bc", "onnx"]:
            # onnx use original input data
            # float bc use original input data
            # quantized bc use original input data
            # quantized bc without desc use runtime data
            if model_info.model_type == "onnx":
                input_names = model_info.sess.input_names
            elif model_info.sess.current_phase == "export":
                input_names = model_info.sess.input_names
            elif model_info.desc:
                preprocessed = False
                input_names = model_info.raw_input_names
            elif not model_info.desc:
                input_names = model_info.sess.input_names
                input_files_batch = 2
                input_file_idx = model_idx
        elif self.model_types == ["bc", "bc"]:
            # float bc use original input data
            # quantized bc use original input data
            # quantized bc without desc use runtime data
            if model_info.sess.current_phase == "export" and \
               not model_info.desc:
                input_names = model_info.sess.input_names
                input_files_batch = 2
                input_file_idx = model_idx
            elif model_info.desc:
                preprocessed = False
                input_names = model_info.raw_input_names
            elif not model_info.desc:
                input_names = model_info.sess.input_names
                input_files_batch = 2
                input_file_idx = model_idx
        elif self.model_types == ["onnx", "onnx"]:
            input_names = model_info.raw_input_names
        else:
            input_names = model_info.sess.input_names

        if len(self.input_files) != input_files_batch:
            raise ValueError("The number of input files batch "
                             f"{len(self.input_files)} invalid.")

        if len(self.input_files[input_file_idx]) != len(input_names):
            raise ValueError("The input data you configured does not"
                             " match the number of model inputs. "
                             f"Model need {len(input_names)} input files, "
                             "while you configured "
                             f"{len(self.input_files[input_files_batch-1])} "
                             "inputs.")

        for _idx, input_file in enumerate(self.input_files[input_file_idx]):
            _model_input_names = deepcopy(input_names)
            _model_input_name = input_names[_idx]

            input_shape = model_info.sess.input_shapes[_idx]
            input_dtype = model_info.sess.input_types[_idx]
            file_path = input_file

            for _input_name in _model_input_names:
                if input_file.startswith(f"{_input_name}:"):
                    # fresh input_names, input_shape, input_dtype
                    file_path = input_file[len(f"{_input_name}:"):]
                    _model_input_name = _input_name
                    input_file_contain_names.append(_model_input_name)
                    input_shape = model_info.sess.input_shapes[
                        _model_input_names.index(_model_input_name)]
                    input_dtype = model_info.sess.input_types[
                        _model_input_names.index(_model_input_name)]
                    break

            if not os.path.isfile(file_path):
                raise ValueError(f'input data does not exist: {file_path}')

            input_info = InputInfo(path=file_path,
                                   name=_model_input_name,
                                   preprocessed=preprocessed)

            # check file format
            if file_path.endswith('npy'):
                input_info.data = np.load(file_path)
            else:
                error_message.append("Only support npy file format.")
                continue
            # check file shape
            if (_model_input_name.endswith(("_y", "_uv")) and input_shape in [
                [1, 8192, 8192, 1],
                [1, 4096, 4096, 2],
                [1, None, None, 1],
                [1, None, None, 2],
            ] and (_model_input_name[:_model_input_name.rfind('_')] +
                   "_y" in input_names
                   and _model_input_name[:_model_input_name.rfind('_')] +
                   "_roi" in input_names) and preprocessed):
                # skip resizer input check
                logging.warning("skip %s resizer input %s check input_file",
                                model_info.model_type, _model_input_name)
            elif list(input_info.data.shape) != list(
                    input_shape) and preprocessed:
                # ignore onnx model input data shape check
                # cause input_batch will change the shape
                if "onnx" not in self.model_types:
                    error_message.append(
                        f"{model_info.model_type} "
                        f"{_model_input_name}: "
                        "input data shape"
                        f" {input_info.data.shape} "
                        f"!= model input shape: {input_shape}")
            # check file dtype
            if input_info.data.dtype != input_dtype and preprocessed:
                error_message.append(f"{model_info.model_type} "
                                     f"{_model_input_name}: input data dtype "
                                     f"{input_info.data.dtype} "
                                     f"!= model input dtype: {input_dtype}")
            inputs_info.append(input_info)

        if input_file_contain_names and set(input_file_contain_names) != set(
                input_names):
            error_message.append(
                f"input file name does not match model input name: "
                f"{input_file_contain_names} vs {input_names}")

        if error_message:
            raise ValueError("Check input files failed: \n" +
                             "\n".join(error_message))

        self.params.inputs_info.append(inputs_info)

        return None

    def check_input(self) -> None:
        if not self.input_files:
            if os.environ.get("HORIZON_TC_UI_DEBUG"):
                return None
            raise ValueError(
                "Input file missing, please specify --input parameter.")
        self.input_files = [i.split(",") for i in self.input_files]

        for idx, model_info in enumerate(self.params.models_info):
            self.check_input_file(idx, model_info)

    def format_input_name(self, input_name: str,
                          model_type: Literal["onnx", "bc", "hbm"]) -> str:
        if model_type == "onnx":
            return input_name

        desc = {
            k.lower(): v
            for k, v in self.params.models_info[1].desc.items()
        }

        if not desc:
            return input_name

        input_batch = desc["input_batch"]
        separate_batch = desc["separate_batch"] == "True"
        input_source = desc["input_source"]

        if input_name in input_source:
            return input_name

        batch = False
        if input_batch and int(input_batch) != 1 and separate_batch:
            batch = True
        if 'separate_name' in desc.keys() and input_name.split(
                '_', 1)[0] in desc['separate_name']:  # noqa
            batch = True

        if not input_name.endswith(('_y', '_uv', '_roi')) and not batch:
            return input_batch

        if input_name.endswith(('_y', '_uv', '_roi')):
            input_name = input_name.replace('_y', '')
            input_name = input_name.replace('_uv', '')
            input_name = input_name.replace('_roi', '')
        if batch and input_name.rfind('_') != -1:
            input_name = input_name[:input_name.rfind('_')]

        return input_name

    def check_model_name(self) -> None:
        sess1 = self.params.models_info[0].sess
        sess2 = self.params.models_info[1].sess
        model1_type = self.params.models_info[0].model_type
        model2_type = self.params.models_info[1].model_type
        desc1 = self.params.models_info[0].desc
        desc2 = self.params.models_info[1].desc
        skip_check = False if desc1 and desc2 else True
        model1_raw_input_names = [
            self.format_input_name(i, model1_type) for i in sess1.input_names
        ]
        for _model2_input_name in sess2.input_names:
            model2_input_name = self.format_input_name(_model2_input_name,
                                                       model2_type)
            if model2_input_name in model1_raw_input_names or skip_check:
                continue

            message = "models input name different. "
            message += f"'{model2_input_name}' not in {model1_raw_input_names}"
            raise ValueError(message)
        model2_raw_input_names = [
            self.format_input_name(i, model2_type) for i in sess2.input_names
        ]

        self.params.models_info[0].raw_input_names = list(
            dict.fromkeys(model1_raw_input_names))
        self.params.models_info[1].raw_input_names = list(
            dict.fromkeys(model2_raw_input_names))
        # If layer_out_dump is configured it modifies the model outputs!!!
        if sorted(sess1.output_names) != sorted(sess2.output_names):
            logging.info("The output names of the two models are different.")
        return None

    def check_model_shape_and_type(self) -> None:

        if "onnx" in self.model_types:
            return None

        error_message = []
        check_input = False

        if self.model_types == ["bc", "hbm"]:
            check_input = True

        input_source_1 = self.params.models_info[0].desc.get(
            "INPUT_SOURCE", {})
        input_source_2 = self.params.models_info[1].desc.get(
            "INPUT_SOURCE", {})

        if "resizer" in input_source_1.values(
        ) or "resizer" in input_source_2.values():
            check_input = False

        input_shapes_1 = self.params.models_info[0].sess.input_shapes
        output_shapes_1 = self.params.models_info[0].sess.output_shapes
        input_types_1 = self.params.models_info[0].sess.input_types
        output_types_1 = self.params.models_info[0].sess.output_types

        output_trans_index = [
            self.params.models_info[1].sess.output_names.index(i)
            for i in self.params.models_info[0].sess.output_names
        ]

        output_types_2 = [
            self.params.models_info[1].sess.output_types[i]
            for i in output_trans_index
        ]
        output_shapes_2 = [
            self.params.models_info[1].sess.output_shapes[i]
            for i in output_trans_index
        ]

        if check_input:
            input_trans_index = [
                self.params.models_info[1].sess.input_names.index(i)
                for i in self.params.models_info[0].sess.input_names
            ]
            input_shapes_2 = [
                self.params.models_info[1].sess.input_shapes[i]
                for i in input_trans_index
            ]
            input_types_2 = [
                self.params.models_info[1].sess.input_types[i]
                for i in input_trans_index
            ]
            if input_types_1 != input_types_2:
                error_message.append(
                    "The input types of the two models are different: "
                    f"{input_types_1} vs {input_types_2}")

            if input_shapes_1 != input_shapes_2:
                error_message.append(
                    "The input shapes of the two models are different: "
                    f"{input_shapes_1} vs {input_shapes_2}")

        if output_shapes_1 != output_shapes_2:
            error_message.append(
                "The output shapes of the two models are different: "
                f"{output_shapes_1} vs {output_shapes_2}")

        if output_types_1 != output_types_2:
            # TODO(wenhao.ma): Add remove dequantize node model support
            error_message.append(
                "The output types of the two models are different: "
                f"{output_types_1} vs {output_types_2}")
        if error_message:
            raise ValueError("Check model shape and type failed: \n" +
                             "\n".join(error_message))

    def add_onnx_tenser_output(self) -> None:
        intermediate_node_info = {}
        for model_info in self.params.models_info:
            if model_info.model_type != "onnx":
                continue
            model = onnx.load(model_info.path)
            output_names = [output.name for output in model.graph.output]
            raw_output_names = deepcopy(output_names)
            for node in model.graph.node:
                for output in node.output:
                    if output in output_names:
                        continue
                    model.graph.output.extend([ValueInfoProto(name=output)])
                    intermediate_node_info.update({output: node.name})
            model = infer_shapes(model)
            sess = HB_ONNXRuntime(onnx_model=model)
            model_info.sess = sess
            model_info.raw_output_names = raw_output_names

    def check_onnx_graph(self):
        for model_info in self.params.models_info:
            if model_info.model_type != "onnx":
                continue
            self.params.output_names = model_info.output_names
            for node in model_info.sess.onnx_model.graph.node:
                for output in node.output:
                    self.params.graph.update({output: node.name})

    def set_model_graph_from_bc(self, idx: int):
        pass
