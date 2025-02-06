# Copyright (c) 2024 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import logging

import numpy as np

from horizon_tc_ui.utils.tool_utils import print_table
from horizon_tc_ui.verifier import VerifierParams


class VerifierComparator:
    def __init__(self, params_info: VerifierParams):
        self.params_info = params_info

    def run(self):
        self.preprocess()
        if self.params_info.mode == "consistency":
            self.consistency()
        elif self.params_info.mode == "cosine":
            self.consine_similarity()

    def preprocess(self):
        pass

    def consistency(self):
        status = None
        for name, output1 in self.params_info.models_info[0].outputs.items():
            if name not in self.params_info.models_info[1].outputs:
                continue

            output2 = self.params_info.models_info[1].outputs[name]
            # convert bool array to int
            output1 = output1.astype(
                np.int) if output1.dtype == np.bool else output1  # noqa
            output2 = output2.astype(
                np.int) if output2.dtype == np.bool else output2  # noqa
            # Compare output1 and output2 with input digits
            ret = np.allclose(output1,
                              output2,
                              atol=10**-self.params_info.digits)
            # compute mismatched elements with format such as 1/2000
            mismatched_elements = np.sum(output1 != output2)
            total_elements = output1.size
            percentage = (mismatched_elements / total_elements) * 100
            mismatched = f"{mismatched_elements}/{total_elements} ({percentage:.2f}%)"  # noqa
            # compute max absolute difference
            max_abs_diff = np.max(np.abs(output1 - output2))
            # compute max relative difference
            max_rel_diff = np.max(
                np.abs(output1 - output2) /
                np.maximum(np.abs(output1), 1e-10))  # noqa
            self.params_info.consistency_info[name] = [
                ret, mismatched, max_abs_diff, max_rel_diff
            ]
            if ret:
                status = True
                continue
            else:
                status = False

        if status is None:
            logging.error("No output to compare")
        if status:
            logging.info("Model output consistency compare success")
        else:
            logging.error("Model output consistency compare failed")

    def check_tensor_name(self, name, idx):
        names = self.params_info.models_info[idx].outputs.keys()
        if name in names:
            return name
        elif name + "_quantized" in names:
            return name + "_quantized"
        elif name + "_calibrated" in names:
            return name + "_calibrated"
        else:
            return None

    def preprocess_of_batch_data(self, name, output1, output2):
        if output1.shape != output2.shape and \
                (output1.size > 1 and output2.size > 1) and \
                (output1.shape[0] == 1 or output2.shape[0] == 1) and \
                (output1.shape[1:] == output2.shape[1:]):
            logging.info(
                f"[{name}] First dimension is different, extended output")
            if output1.shape[0] == 1:
                output1 = np.tile(output1, (output2.shape[0], ) + (1, ) *
                                  (output1.ndim - 1))  # noqa
            else:
                output2 = np.tile(output2, (output1.shape[0], ) + (1, ) *
                                  (output2.ndim - 1))  # noqa
        return output1, output2

    def preprocess_of_data_layout(self, output1, output2):
        if output1.ndim != 4 or output2.ndim != 4:
            return output1, output2

        if output1.shape == output2.shape:
            return output1, output2

        # NCHW vs NHWC
        if output1.shape[0] == output2.shape[0] and \
                output1.shape[1] == output2.shape[3] and \
                output1.shape[2] == output2.shape[1] and \
                output1.shape[3] == output2.shape[2]:
            # NHWC -> NCHW
            output2 = output2.transpose(0, 3, 1, 2)
        # NHWC vs NCHW
        if output1.shape[0] == output2.shape[0] and \
                output1.shape[1] == output2.shape[2] and \
                output1.shape[2] == output2.shape[3] and \
                output1.shape[3] == output2.shape[1]:
            # NCHW -> NHWC
            output2 = output2.transpose(0, 2, 3, 1)
        return output1, output2

    def consine_similarity(self):
        for name, output1 in self.params_info.models_info[0].outputs.items():
            target_name = self.check_tensor_name(name, 1)
            if not target_name:
                continue

            output2: np.ndarray = self.params_info.models_info[1].outputs[
                target_name]

            if output1.size == 0 or output2.size == 0:
                logging.warning(
                    "output of %s is empty, skip this tensor calculation.",
                    name)
                continue

            # e.g. 1x224x224x3 vs 8x224x224x3
            output1, output2 = self.preprocess_of_batch_data(
                name, output1, output2)
            output1, output2 = self.preprocess_of_data_layout(output1, output2)

            # e.g. 8x224x224x3 vs 8x512x512x3
            if output1.shape != output2.shape:
                logging.warning(
                    f"[{name}] Different shape, skip this tensor calculation."
                    f"model shape: {output1.shape} vs {output2.shape}")
                continue

            if np.all(output1 == 0) and np.all(output2 == 0):
                self.params_info.cosine_info[name] = 1.0
                continue

            dot_product = np.dot(output1.flatten(), output2.flatten())
            norm_v1 = np.linalg.norm(output1.flatten())
            norm_v2 = np.linalg.norm(output2.flatten())
            if norm_v1 * norm_v2 == 0:
                logging.warning("tensor %s consine compute error.", name)
                logging.debug("output1 model path: %s",
                              self.params_info.models_info[0].path)  # noqa
                logging.debug("[ L2 norm ] max: %s, %s", norm_v1.max(),
                              norm_v1.min())  # noqa
                logging.debug("output2 model path: %s",
                              self.params_info.models_info[1].path)  # noqa
                logging.debug("[ L2 norm ] max: %s, %s", norm_v2.max(),
                              norm_v2.min())  # noqa
                continue
            consine_similarity = dot_product / (norm_v1 * norm_v2)

            self.params_info.cosine_info[name] = f"{consine_similarity:.6g}"

            existing_tensor = {
                k: self.params_info.cosine_info[k]
                for k in self.params_info.graph
                if k in self.params_info.cosine_info
            }  # noqa
            non_existing_tensor = {
                k: v
                for k, v in self.params_info.cosine_info.items()
                if k not in self.params_info.graph
            }  # noqa

            self.params_info.cosine_info = {
                **non_existing_tensor,
                **existing_tensor
            }

    def print_consine_table(self) -> None:
        data = [[
            'NodeName', 'TensorName', "ConsineSimilarity"
        ]] if self.params_info.graph else [['TensorName', "ConsineSimilarity"]]

        output_data = []
        for k, v in self.params_info.cosine_info.items():
            if not self.params_info.graph:
                if k not in self.params_info.output_names:
                    data.append([k, v])
                else:
                    output_data.append([k, v])
                continue
            if k not in self.params_info.graph:
                continue
            if k not in self.params_info.output_names:
                data.append([self.params_info.graph[k], k, v])
            else:
                output_data.append([self.params_info.graph[k], k, v])
        data.extend(output_data)
        print_table(data)

    def print_consistency_table(self) -> None:
        data = [[
            'OutputName', "Consistency", "Mismatched Elements", "Max Abs Diff",
            "Max Rel Diff"
        ]]

        for output_name, consistency_info in self.params_info.consistency_info.items():  # noqa yapf: disable
            data.append([output_name, *[str(i) for i in consistency_info]])
        print_table(data)

    def print(self) -> None:
        if self.params_info.mode == "cosine":
            self.print_consine_table()
        else:
            self.print_consistency_table()
