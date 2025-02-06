# Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import logging
import os
from copy import deepcopy
from typing import Any, Dict, Literal, Union

import click
import yaml

from horizon_tc_ui import HBRuntime
from horizon_tc_ui import __file__ as tc_ui_root_file
from horizon_tc_ui.config.mapper_consts import march_list
from horizon_tc_ui.config.schema_yaml import schema_yaml
from horizon_tc_ui.parser.caffe_parser import CaffeParser
from horizon_tc_ui.utils.tool_utils import (get_str_from_list,
                                            init_root_logger,
                                            on_exception_exit)


class ConfigGenerator:
    """Generate simple/full/custom_op template yaml
    """
    def __init__(self) -> None:
        tc_ui_path = os.path.abspath(os.path.dirname(tc_ui_root_file))
        self.template_yaml_path = os.path.join(tc_ui_path, 'template')
        self.custom_op_path = os.path.join(self.template_yaml_path,
                                           'sample_custom.py')
        self.current_path = os.path.abspath(os.getcwd())
        self.schema_yaml = self.parse_schema_yaml()

    def get_intersection_from_schema_dict(
            self, schema_dict: dict,
            template_dict: dict) -> Dict[str, Dict[str, Any]]:
        """Find the intersection of schema_dict and template_dict,
        and assign the value of template_dict to the intersection.

        Args:
            schema_dict (dict): schema_yaml dict
            template_dict (dict): template yaml dict

        Raises:
            ValueError: if key from template_dict is not in schema_dict

        Returns:
            Dict[str, Dict[str, Any]]: intersection dict
        """
        updated_dict = {}
        for key in template_dict:
            if key in schema_dict:
                if isinstance(template_dict[key], dict) and isinstance(
                        schema_dict[key], dict):
                    updated_subdict = self.get_intersection_from_schema_dict(
                        schema_dict[key], template_dict[key])
                    if updated_subdict:
                        updated_dict[key] = updated_subdict
                else:
                    updated_dict[key] = template_dict[key]
            else:
                raise ValueError(f'key {key} not in schema_yaml')
        return updated_dict

    def parse_schema_yaml(self) -> Dict[str, Dict[str, Any]]:
        """Parse horizon_tc_ui.config.schema_yaml to pure dict

        Returns:
            dict: full schema dict
        """
        parsed_dict = {}
        for group_name, group_params in schema_yaml.items():
            parsed_dict[group_name] = {}
            for param in group_params:
                param_key = getattr(param, 'key', None)
                param_value = getattr(param, 'default', None)
                if param_key:
                    parsed_dict[group_name][param_key] = param_value
        return parsed_dict

    def read_template(
        self, template_type: Literal['simple', 'full', 'fast_perf']
    ) -> Dict[str, Dict[str, Any]]:
        """Read template yaml by template name

        Args:
            template_type (Literal[ &#39;simple&#39;, &#39;full&#39;, &#39;fast_perf&#39;]): template yaml name

        Returns:
            Dict[str, Dict[str, Any]]: parsed yaml dict
        """  # noqa
        template_path = os.path.join(self.template_yaml_path,
                                     f"{template_type}_template.yaml")
        with open(template_path, 'r', encoding='utf-8') as f:
            template = yaml.safe_load(f)
        schema_dict = self.parse_schema_yaml()
        res = self.get_intersection_from_schema_dict(schema_dict, template)
        return res

    def get_model_sess(self, model_file: str,
                       model_proto: str) -> Union[HBRuntime, CaffeParser]:
        """Get model runtime

        Args:
            model_file (str): model file
            model_proto (str): caffe model proto

        Raises:
            ValueError: model not in onnx/caffe/bc

        Returns:
            Union[HBRuntime, CaffeParser]: onnx bc use HBRuntime, caffe use CaffeParser
        """  # noqa
        if model_proto:
            sess = CaffeParser(model_file=model_file, model_proto=model_proto)
        elif model_file.endswith(('.onnx', '.bc')):
            sess = HBRuntime(model_file)
        else:
            raise ValueError(f'Unsupport model: {model_file}')
        return sess

    def update_yaml_repeat_params(self,
                                  yaml_dict: dict,
                                  repeat_param_mapping: dict,
                                  repeat_num: int = 1) -> None:
        """Update repeat params value by repeat_num

        Args:
            yaml_dict (dict): yaml dict
            repeat_param_mapping (str): mapping of repeat params
            repeat_num (int, optional): repeat_num. Defaults to 1.
        """
        for group_name, group_params in repeat_param_mapping.items():
            for param_name in group_params:
                _value = yaml_dict[group_name].get(param_name)
                if _value:
                    yaml_dict[group_name][param_name] = ";".join(
                        [str(_value)] * repeat_num)

    def update_by_model(self, src: dict, model_file: str,
                        model_proto: str) -> Dict[str, Dict[str, Any]]:
        """Update model info to src dict

        Args:
            src (dict): source dict
            model_file (str): onnx/caffe/bc model file
            model_proto (str): caffe proto file

        Raises:
            ValueError: Only support onnx/caffe/bc model

        Returns:
            dict: dict updated model info
        """
        result = deepcopy(src)
        sess = self.get_model_sess(model_file=model_file,
                                   model_proto=model_proto)
        if model_proto:
            result["model_parameters"]["caffe_model"] = os.path.relpath(
                model_file)
            result["model_parameters"]["prototxt"] = os.path.relpath(
                model_proto)
            if "onnx_model" in result['model_parameters']:
                del result['model_parameters']["onnx_model"]
        elif model_file.endswith('.onnx'):
            result["model_parameters"]["onnx_model"] = os.path.relpath(
                model_file)
            if "caffe_model" in result['model_parameters']:
                del result['model_parameters']["caffe_model"]
            if "prototxt" in result['model_parameters']:
                del result['model_parameters']["prototxt"]
        elif model_file.endswith('.bc'):
            sess = HBRuntime(model_file)
        else:
            raise ValueError(f'Unsupport model: {model_file}')

        if len(sess.input_names) == 1:
            input_batch = sess.input_shapes[0][0]
            if input_batch in ['?', '-1', '0']:
                input_batch = 1
        else:
            input_batch = None

        model_prefix = os.path.splitext(os.path.basename(model_file))[0]

        input_names = sess.input_names
        input_num = len(input_names)
        input_shapes = [
            "x".join(
                [str(i) if i not in ['?', '-1', '0'] else '1' for i in _list])
            for _list in sess.input_shapes
        ]

        if result['input_parameters'].get('norm_type'):
            norm_type = [
                result['input_parameters']['norm_type']
                if _layout else "no_preprocess"
                for _layout in sess.input_layouts
            ]
            result['input_parameters']['norm_type'] = get_str_from_list(
                norm_type)

        input_type_train = [
            "bgr" if _layout else "featuremap"
            for _layout in sess.input_layouts
        ]
        input_type_rt = [
            "nv12" if _layout else "featuremap"
            for _layout in sess.input_layouts
        ]
        layout = [
            _layout if _layout != '' else 'NCHW'
            for _layout in sess.input_layouts
        ]

        result['input_parameters']['input_name'] = get_str_from_list(
            input_names)
        result['input_parameters']['input_shape'] = get_str_from_list(
            input_shapes)
        result['input_parameters']['input_layout_train'] = get_str_from_list(
            layout)
        result['input_parameters']['input_type_train'] = get_str_from_list(
            input_type_train)
        result['input_parameters']['input_type_rt'] = get_str_from_list(
            input_type_rt)

        # TODO Adapt input_batch change, default is 1.
        if input_batch:
            result['input_parameters']['input_batch'] = input_batch

        result['model_parameters']['output_model_file_prefix'] = model_prefix

        repeat_param_mapping = {
            "calibration_parameters": ['cal_data_dir'],
            "input_parameters": ['mean_value', 'scale_value']
        }

        self.update_yaml_repeat_params(result, repeat_param_mapping, input_num)
        return result

    def save(self, template: dict, generate_type: str) -> str:
        """Save yaml by generate_type

        Args:
            template (dict): template yaml dict
            generate_type (str): generate yaml type

        Returns:
            str: reslut yaml save path
        """
        dst_name = f"{generate_type}_compile_config.yaml"
        if 'yaml' in generate_type:
            logging.warning(
                'You need to configure the march and model parameters in %s',
                dst_name)
            logging.warning(
                'Currently, both onnx and caffe models are not supported, '
                'please choose one to configure')
        dst = os.path.join(self.current_path, dst_name)
        with open(dst, 'w', encoding='utf-8') as f:
            yaml.safe_dump(template, f)
        logging.info('Successfully generate %s', dst)
        return dst

    def generate(self,
                 generate_type: Literal['custom_op', 'simple', 'full',
                                        'fast_perf'],
                 model: str = "",
                 proto: str = "",
                 march: str = "") -> None:

        if generate_type == 'custom_op':
            logging.warning('The custom_op option is not supported now.')
            return
        else:
            template = self.read_template(generate_type)

        if model:
            template = self.update_by_model(template, model, proto)
        elif not model and proto:
            raise ValueError("Please set --model option")

        if march in march_list:
            template["model_parameters"]["march"] = march

        self.save(template, generate_type)


@click.command()
@click.help_option('--help', '-h')
@click.option('-c',
              '--custom-op',
              is_flag=True,
              help='Generate custom op template file',
              default=False,
              hidden=True)
@click.option('-s',
              '--simple-yaml',
              is_flag=True,
              help='Generate simple yaml template file',
              default=False)
@click.option('-f',
              '--full-yaml',
              is_flag=True,
              help='Generate full yaml template file',
              default=False)
@click.option('-m', '--model', help='Model to be compiled')
@click.option('-p', '--proto', help='Caffe model proto file')
@click.option('--march', help='BPU micro arch', type=click.Choice(march_list))
@on_exception_exit
def cmd_main(custom_op: bool, simple_yaml: bool, full_yaml: bool, model: str,
             proto: str, march: str) -> None:
    log_level = logging.DEBUG
    init_root_logger('hb_config_generator', file_level=log_level)
    if not custom_op and not simple_yaml and not full_yaml:
        raise ValueError("Please select an option to run, "
                         "use hb_config_generator -h too see options")
    if sorted([custom_op, simple_yaml, full_yaml]) != [False, False, True]:
        raise ValueError("Please make sure only one option is turned on")
    if custom_op:
        generate_type = 'custom_op'
    elif simple_yaml:
        generate_type = 'simple'
    elif full_yaml:
        generate_type = 'full'

    config_generator = ConfigGenerator()
    config_generator.generate(generate_type=generate_type,
                              model=model,
                              proto=proto,
                              march=march)


if __name__ == "__main__":
    cmd_main()
