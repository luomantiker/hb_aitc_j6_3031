# Copyright (c) 2022 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import logging

from horizon_tc_ui.hb_hbmruntime import HB_HBMRuntime
from horizon_tc_ui.verifier.params_check import VerifierParams


class VerifierInference:
    def __init__(self, params_info: VerifierParams) -> None:
        self.params_info = params_info

    def run(self) -> None:
        for idx, model_info in enumerate(self.params_info.models_info):
            logging.info(f"{model_info.name} inference...")
            if model_info.model_type == 'onnx':
                self.onnx_inference(idx)
            elif model_info.model_type == 'bc':
                self.hbir_inference(idx)
            elif model_info.model_type == 'hbm':
                self.hbm_inference(idx)

    def onnx_inference(self, idx: int) -> None:
        model_info = self.params_info.models_info[idx]

        output_names = model_info.sess.output_names
        outputs = model_info.sess.run(output_names, model_info.inputs)
        for idx, output_name in enumerate(output_names):
            model_info.outputs.update({output_name: outputs[idx]})

        return None

    def hbir_inference(self, idx: int) -> None:
        model_info = self.params_info.models_info[idx]

        for input_idx, input_name in enumerate(model_info.sess.input_names):
            dtype = model_info.sess.input_types[input_idx]
            model_info.inputs[input_name] = model_info.inputs[
                input_name].astype(dtype)

        output_names = model_info.sess.output_names
        outputs = model_info.sess.run(output_names, model_info.inputs)
        for idx, output_name in enumerate(output_names):
            model_info.outputs.update({output_name: outputs[idx]})

        for node_info, result in model_info.sess.intermediate_outputs.items():
            tensor_name = node_info.split("__")[1]
            model_info.outputs.update({tensor_name: result})
        return None

    def hbm_inference(self, idx: int) -> None:
        model_info = self.params_info.models_info[idx]
        sess: HB_HBMRuntime = model_info.sess

        for input_idx, input_name in enumerate(sess.input_names):
            dtype = sess.input_types[input_idx]
            model_info.inputs[input_name] = model_info.inputs[
                input_name].astype(dtype)  # noqa

        output_names = sess.output_names
        outputs = sess.run(output_names, model_info.inputs)
        for idx, output_name in enumerate(output_names):
            model_info.outputs.update({output_name: outputs[idx]})
        return None
