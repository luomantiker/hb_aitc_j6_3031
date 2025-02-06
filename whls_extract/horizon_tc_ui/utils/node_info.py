# Copyright (c) 2024 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import json
import logging
import os
import shutil

import numpy as np

from horizon_tc_ui.config.config_info import ConfigInfo
from horizon_tc_ui.hb_verifier import verifier
from horizon_tc_ui.utils.tool_utils import dump_table, format_str, print_table


class NodeInfo:
    def __init__(self, conf: ConfigInfo):
        self.conf = conf
        prefix = conf.output_model_file_prefix_full
        self.quant_info_path = prefix + "_quant_info.json"
        self.advice_path = prefix + "_advice.json"
        self.quant_info = {}
        self.advice_info = {}
        self.mode = "skip" if not self.conf.first_cali_data else "full"
        self.node_info = []
        self.tensor_info = []
        self.params_info = None
        self.init_info_header()

    def init_info_header(self):
        self.tensor_info.append(
            ["TensorName", "Calibrated Cosine", "Quantized Cosine"])
        if self.mode == 'skip':
            self.node_info.append(
                ['Node', 'NodeType', 'ON', 'Output Data Type'])
        else:
            self.node_info.append([
                'Node', 'NodeType', 'ON', 'Threshold', 'Calibrated Cosine',
                'Quantized Cosine', 'Output Data Type'
            ])

    def parser_quant_info(self):
        if not os.path.exists(self.quant_info_path):
            logging.error("quant_info.json is not found.")
            return None

        with open(self.quant_info_path, 'r', encoding='utf-8') as file:
            self.quant_info = json.load(file)

    def parser_advice_info(self):
        if not os.path.exists(self.advice_path):
            logging.error("advice.json %s is not found.", self.advice_path)
            return None

        with open(self.advice_path, 'r', encoding='utf-8') as file:
            _advice_info = json.load(file)

        for info in _advice_info:
            if 'tensor_names' not in info:
                logging.warning(f"'tensor_names' is not found in {info}")
                continue
            if len(info['tensor_names']) == 0:
                logging.warning(f"'tensor_names' is empty in {info}")
                continue
            if 'tensor_name' not in info['tensor_names'][0]:
                logging.warning(f"'tensor_name' is not found in {info}")
                continue
            tensor_name = info['tensor_names'][0]['tensor_name']
            data_type = info['tensor_names'][0].get('data_type', '--')
            self.advice_info.update(
                {tensor_name: {
                    'data_type': data_type,
                    'advice': info
                }})

    def search_real_info(self, node_info):
        if 'inputs' not in node_info:
            logging.warning(f"'inputs' is not found in {node_info}")
            return None
        if len(node_info['inputs']) == 0:
            logging.warning(f"'inputs' is empty in {node_info}")
            return None

        input_tensor_name = node_info['inputs'][0]
        for node_name, node_info in self.quant_info.items():
            if node_name == 'model_output_tensor':
                continue
            if node_info['type'] != 'Conv' or not node_info['outputs']:
                continue
            if node_info['outputs'][0] == input_tensor_name:
                return [node_name, node_info]
        return None

    def get_display_node_name(self, node_name, node_info):
        related_info = self.search_real_info(node_info)
        if not related_info:
            return format_str(node_name)
        return format_str(related_info[0]) + "+" + format_str(node_name)

    def get_thresholds(self, node_name):
        node_info = self.quant_info[node_name]
        if 'thresholds' not in node_info:
            logging.warning(f"'thresholds' is not found in {node_info}")
            return '--'
        if len(node_info['thresholds']) != 0 \
                and len(node_info['thresholds'][0]) != 0:
            return node_info['thresholds'][0][0]

        if 'type' not in node_info:
            logging.warning(f"'type' is not found in {node_info}")
            return '--'
        if node_info['type'] not in ['Add', 'Clip', 'Relu']:
            logging.info(f"node {node_name} is not Add, Clip or Relu "
                         "and has no thresholds")
            return '--'

        related_info = self.search_real_info(node_info)
        if not related_info:
            return '--'

        _, real_node_info = related_info
        if 'thresholds' not in real_node_info:
            return '--'
        if len(real_node_info['thresholds']) == 0 or \
                len(real_node_info['thresholds'][0]) == 0:
            return '--'
        return real_node_info['thresholds'][0][0]

    def get_tensor_name(self, node_name):
        node_info = self.quant_info[node_name]
        return node_info['outputs'][0]

    def get_advice_name(self, tensor_name):
        if tensor_name in self.advice_info:
            return tensor_name
        if tensor_name not in self.advice_info and \
                tensor_name + '_quantized' in self.advice_info:
            tensor_name += '_quantized'
            return tensor_name
        if tensor_name not in self.advice_info and \
                tensor_name + '_calibrated' in self.advice_info:
            tensor_name += '_calibrated'
            return tensor_name
        return None

    def ignore_node(self, node_name) -> bool:
        node_info = self.quant_info[node_name]
        if 'outputs' not in node_info or len(node_info['outputs']) == 0:
            return True

        tensor_name = node_info["outputs"][0]
        for node_name, node_info in self.quant_info.items():
            if node_name == "model_output_tensor":
                continue
            if "inputs" not in node_info or "type" not in node_info or \
                    "thresholds" not in node_info:
                continue
            if len(node_info["inputs"]) == 0 or \
                    len(node_info["thresholds"]) == 0:
                continue

            if node_info["inputs"][0] == tensor_name and \
                    node_info["type"] in ['Add', 'Clip', 'Relu'] and \
                    len(node_info["thresholds"][0]) == 0:
                return True

        return False

    def get_node_type(self, node_name):
        node_info = self.quant_info[node_name]
        if 'type' not in node_info:
            logging.warning(f"'type' is not found in {node_info}")
            return '--'

        if 'thresholds' not in node_info:
            return node_info['type']
        if len(node_info['thresholds']) != 0 \
                and len(node_info['thresholds'][0]) != 0:
            return node_info['type']

        if node_info['type'] not in ['Add', 'Clip', 'Relu']:
            return node_info['type']

        related_info = self.search_real_info(node_info)
        if not related_info:
            return node_info['type']

        _, real_node_info = related_info
        if 'type' not in real_node_info:
            return node_info['type']

        return real_node_info['type'] + "+" + node_info['type']

    def get_quant_info_cosine(self, node_name):
        node_info = self.quant_info[node_name]
        if 'cosine_similarity' not in node_info:
            logging.warning(
                f"'cosine_similarity' is not found in {node_info}")  # noqa
            return '--'
        return node_info['cosine_similarity']

    def get_full_process_cosine(self, node_name):
        tensor_name = self.get_tensor_name(node_name)
        if tensor_name not in self.params_info.cosine_info:
            return '--'
        return self.params_info.cosine_info[tensor_name]

    def get_data_type(self, node_name):
        tensor_name = self.get_tensor_name(node_name)
        advice_name = self.get_advice_name(tensor_name)

        if not advice_name:
            # logging.warning(f"{tensor_name} is not found in advice_info")
            return '--'

        return self.advice_info[advice_name]['data_type']

    def get_backend(self, node_name):
        tensor_name = self.get_tensor_name(node_name)
        advice_name = self.get_advice_name(tensor_name)

        if not advice_name:
            # logging.warning(f"{tensor_name} is not found in advice_info")
            return '--'

        backend = '--'
        if 'advice' in self.advice_info[advice_name] and \
                'backend' in self.advice_info[advice_name]['advice']:
            backend = self.advice_info[advice_name]['advice']['backend']
        if backend in ['Vpu', 'Spu']:
            backend = 'BPU'
        if backend == 'external_cpu':
            backend = 'CPU'
        if backend == 'external_jit':
            backend = 'JIT'
        return backend.upper()

    def compute_cosine_similarity(self):
        if len(self.conf.first_cali_data) == 0:
            return None

        prefix_path = self.conf.output_model_file_prefix_full
        optim_model_path = prefix_path + "_optimized_float_model.onnx"
        quanti_model_path = prefix_path + "_quantized_model.bc"
        if not os.path.exists(optim_model_path) or not os.path.exists(
                quanti_model_path):  # noqa
            logging.error("Model Generation Exception.")
            return None

        model = optim_model_path + "," + quanti_model_path
        workspace = self.conf.working_dir + '/.hb_compile_tmp/'
        if os.path.exists(workspace):
            shutil.rmtree(workspace)
        os.mkdir(workspace)

        cali_data_path = []
        for input_name, cali_data in self.conf.first_cali_data.items():
            file_path = workspace + input_name + ".npy"
            np.save(file_path, cali_data)
            cali_data_path.append(file_path)
        parsed_cali_data_path = [
            f"{self.conf.input_names[idx]}:{data_path}"
            for idx, data_path in enumerate(cali_data_path)
        ]  # noqa
        logging.debug("hb_verifier -m %s -i %s", model,
                      ",".join(parsed_cali_data_path))
        self.params_info = verifier(
            model=model, input_files=[",".join(parsed_cali_data_path)])

    def generate_info(self):
        self.parser_quant_info()
        self.parser_advice_info()
        self.compute_cosine_similarity()
        self.generate_tensor_info()
        self.generate_node_info()

    def generate_tensor_info(self):
        if "model_output_tensor" not in self.quant_info:
            return None

        quant_output_tensors = self.quant_info["model_output_tensor"]
        for output_tensor_name, info in quant_output_tensors.items():
            cosine1 = '--'
            if "cosine_similarity" in info:
                cosine1 = info["cosine_similarity"]
            cosine2 = '--'
            if output_tensor_name in self.params_info.cosine_info:
                cosine2 = self.params_info.cosine_info[output_tensor_name]

            self.tensor_info.append([output_tensor_name, cosine1, cosine2])

    def generate_node_info(self):
        for node_name, node_info in self.quant_info.items():
            if node_name == "model_output_tensor":
                continue
            if "outputs" not in node_info:
                logging.warning(
                    f"'outputs' for {node_name} not found in node_info.")
                continue
            if self.ignore_node(node_name):
                continue

            display_node_name = self.get_display_node_name(
                node_name, node_info)  # noqa
            node_type = self.get_node_type(node_name)
            backend = self.get_backend(node_name)
            data_type = self.get_data_type(node_name)

            if self.mode == 'skip':
                self.node_info.append(
                    [display_node_name, node_type, backend, data_type])
                continue

            thresholds = self.get_thresholds(node_name)
            quant_info_cosine = self.get_quant_info_cosine(node_name)
            full_process_cosine = self.get_full_process_cosine(node_name)

            self.node_info.append([
                display_node_name, node_type, backend, thresholds,
                quant_info_cosine, full_process_cosine, data_type
            ])

    def print_node_info(self):
        print_table(self.node_info, logging.info)

    def dump_node_info(self):
        prefix_path = self.conf.output_model_file_prefix_full
        path = prefix_path + "_node_info.csv"
        dump_table(self.node_info, path)

    def print_output_tensor_info(self):
        print_table(self.tensor_info, logging.info)
