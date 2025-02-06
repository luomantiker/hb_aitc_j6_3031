# Copyright (c) 2024 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import logging
from typing import List

from google.protobuf import text_format

from horizon_tc_ui.helper import ModelProtoBase
from horizon_tc_ui.parser.horizon_caffe_pb2 import NetParameter as CaffeNet


class CaffeProto(ModelProtoBase):
    def __init__(self, prototxt) -> None:
        super(CaffeProto, self).__init__()
        self._input_parsed = False
        self.prototxt = prototxt
        self.proto = CaffeNet()
        self.input_name = []
        self.input_dim = {}
        with open(prototxt, 'r') as f:
            text_format.Merge(f.read(), self.proto)

    def parse_input(self) -> None:
        if self._input_parsed is True:
            return
        self._input_parsed = True
        layers = self.proto.layers or self.proto.layer
        # some caffe prototxt define the input layer by layer{} specification,
        # not input
        if len(self.proto.input) == 0:
            for layer in layers:
                if layer.type == 'Input':
                    input_dim = [dim for dim in layer.input_param.shape[0].dim]
                    if len(layer.top) != 1:
                        raise ValueError(
                            "The number of top of Input layer {} is not 1.".
                            format(layer.name))
                    self.input_name.append(layer.top[0])
                    self.input_dim[layer.top[0]] = input_dim
                else:
                    break
        elif len(self.proto.input) == 1:
            self.input_name = self.proto.input
            if len(self.proto.input_dim):
                self.input_dim[self.input_name[0]] \
                    = list(map(int, self.proto.input_dim))
            elif len(self.proto.input_shape):
                self.input_dim[self.input_name[0]] = list(
                    map(int, self.proto.input_shape[0].dim))
            else:
                self.input_dim = {}
                logging.warning("can't get input shape in prototxt.")
        # for same models(op tests), there may be two input layers(eltwise)
        # or more input layers(concat).
        else:
            self.input_name = self.proto.input
            if len(self.proto.input_dim):
                for i in range(len(self.proto.input)):
                    self.input_dim[self.input_name[i]] \
                        = list(map(int, self.proto.input_dim[i]))
            elif len(self.proto.input_shape):
                for i in range(len(self.proto.input)):
                    self.input_dim[self.input_name[i]] \
                        = list(map(int, self.proto.input_shape[i].dim))
            else:
                logging.warning("can't find input size.")

    def get_input_names(self) -> list:
        self.parse_input()
        return [input_name for input_name in self.input_name]

    def get_input_dims(self, input_name) -> list:
        self.parse_input()
        return self.input_dim.get(input_name, [])

    def get_input_shape(self, input_name) -> list:
        return self.get_input_dims(input_name)

    def input_num(self) -> int:
        return len(self.get_input_names())


class CaffeParser:
    def __init__(self, model_file: str, model_proto: str) -> None:
        self.model_file = model_file
        self.model_proto = model_proto
        self.parser = CaffeProto(prototxt=self.model_proto)
        self._input_names = []
        self._input_shapes = []
        self._layout = []

    @property
    def input_names(self) -> List[str]:
        if self._input_names:
            return self._input_names
        self._input_names = self.parser.get_input_names()
        return self._input_names

    @property
    def input_shapes(self) -> List[List[int]]:
        if self._input_shapes:
            return self._input_shapes
        self._input_shapes = [
            self.parser.get_input_shape(input_name)
            for input_name in self.input_names
        ]
        return self._input_shapes

    @property
    def input_layouts(self) -> List[str]:
        if self._layout:
            return self._layout
        for shape_item in self.input_shapes:
            if len(shape_item) != 4:
                self._layout.append("")
                continue
            elif shape_item[0] != 1:
                self._layout.append("")
                continue
            elif shape_item[1] != 3 and shape_item[3] != 3:
                self._layout.append("")
                continue
            elif shape_item[1] == 3 and (shape_item[2] % 2 == 0
                                         and shape_item[3] % 2 == 0):  # noqa
                self._layout.append("NCHW")
                continue
            elif shape_item[3] == 3 and (shape_item[1] % 2 == 0
                                         and shape_item[2] % 2 == 0):  # noqa
                self._layout.append("NHWC")
                continue
            else:
                self._layout.append("")
        return self._layout

    @property
    def layout(self) -> str:
        return self.input_layouts
