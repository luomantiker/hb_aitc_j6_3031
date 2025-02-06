# Copyright (c) 2024 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import logging

import onnx

from horizon_tc_ui.helper import ModelProtoBase


class OnnxModel(ModelProtoBase):
    def __init__(self, model_file=None, model=None):
        super(OnnxModel, self).__init__()
        if model:
            self.model = model
        else:
            self.model_file = model_file
            self.model = onnx.load(model_file)
        self.input = []
        self.input_dim = {}
        self._is_input_parsed = False

    def parse_input(self):
        if self._is_input_parsed:
            return
        self._is_input_parsed = True
        init_names = [init.name for init in self.model.graph.initializer]
        self.input = [
            i for i in self.model.graph.input if i.name not in init_names
        ]
        if len(self.input) == 0:
            raise ValueError("can't find input in model.")

        logging.debug("Model input names: {}".format(
            [i.name for i in self.input]))

        for i in self.input:
            try:
                dim = [d.dim_value for d in i.type.tensor_type.shape.dim]
            except ValueError as e:
                logging.warning(e)
                dim = []
            self.input_dim[i.name] = dim

    def get_input_names(self):
        self.parse_input()
        return [i.name for i in self.input]

    def get_input_dims(self, input_name):
        self.parse_input()
        return self.input_dim.get(input_name, [])

    def get_input_shape(self, input_name):
        return self.get_input_dims(input_name)

    def set_model_batch_dim(self, batch_size=1):
        input_names = self.get_input_names()
        for i in range(len(self.model.graph.input)):
            if self.model.graph.input[i].name in input_names:
                self.model.graph.input[i].type.tensor_type.shape.dim[
                    0].dim_value = batch_size
        self._is_input_parsed = False

    def get_model(self):
        return self.model

    def input_num(self):
        return len(self.get_input_names())

    def search_node(self, node):
        for node_index, node_item in enumerate(self.model.graph.node):
            if node_item.name == node:
                return node_index
        return -1

    def get_node_input_num(self, node):
        for node_item in self.model.graph.node:
            if node_item.name == node:
                return len(node_item.input)
        raise ValueError(f'Node name {node} does not exist in the model')
