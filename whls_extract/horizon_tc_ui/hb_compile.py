# Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import logging
import os
import shutil
from typing import Tuple

import click
from hbdk4.compiler import load
from hbdk4.compiler.version import VERSION as hbdk_version
from horizon_nn.api import version as horizon_nn_version
from onnx.onnx_pb import ModelProto

from horizon_tc_ui.compile.hbm_builder import HBMBuilder
from horizon_tc_ui.compile.ptq_model_builder import PTQModelBuilder
from horizon_tc_ui.config.config_info import ConfigInfo
from horizon_tc_ui.config.mapper_consts import march_list
from horizon_tc_ui.config.params_parser import ParamsParser
from horizon_tc_ui.hbir_handle import HBIRHandle
from horizon_tc_ui.utils.tool_utils import (init_root_logger,
                                            on_exception_exit, update_yaml)
from horizon_tc_ui.utils.yaml_builder import YamlBuilder
from horizon_tc_ui.version import __version__ as horizon_tc_ui_version


def params_check(yaml_path: str) -> ConfigInfo:
    logging.info('Start verifying yaml')
    params_parser = ParamsParser(yaml_path=yaml_path)
    params_parser.validate_parameters()
    logging.info('End verifying yaml')
    return params_parser.conf


def copy_compile_log_file(working_dir: str) -> None:
    log_path = os.path.join(os.getcwd(), 'hb_compile.log')
    if not os.path.exists(log_path) or not os.path.exists(working_dir):
        return None

    destination_log_path = os.path.join(working_dir, 'hb_compile.log')
    if log_path == destination_log_path:
        return None
    if os.path.exists(destination_log_path):
        os.remove(destination_log_path)
    shutil.copy(log_path, destination_log_path)


def ptq_model_build(conf: ConfigInfo) -> ModelProto:
    ptq_builder = PTQModelBuilder(
        model=conf.load_model,
        march=conf.march,
        conf=conf,
        name_prefix=conf.output_model_file_prefix_full)
    ptq_builder.build()

    return ptq_builder.calibration_model


def fast_perf_handle(
        model: str, proto: str, march: str,
        input_shape: Tuple[Tuple[str, str]]) -> Tuple[ConfigInfo, ModelProto]:
    if not march:
        raise ValueError("fast-perf mode is turned on, "
                         f"please specify march oiption from {march_list}")

    if not os.path.isfile(str(model)):
        raise ValueError(f'user input model is not a file: {model}')

    if model.endswith('.onnx') and not proto:
        model_type = 'onnx'
    elif not model.endswith('.onnx') and proto:
        model_type = 'caffe'
    else:
        raise ValueError('fast-perf currently only supports '
                         'onnx and caffe models.')

    yaml_path = YamlBuilder(mode="fast_perf",
                            model=model,
                            proto=proto,
                            model_type=model_type,
                            march=march,
                            input_shape=input_shape).build()
    conf = params_check(yaml_path=yaml_path)
    ptq_model = ptq_model_build(conf)
    return conf, ptq_model


def model_check_handle(
        model: str, proto: str, march: str,
        input_shape: Tuple[str, str]) -> Tuple[ConfigInfo, ModelProto]:
    if not march:
        raise ValueError(f"Please specify march option from {march_list}")
    if not model.endswith(('onnx', 'caffemodel', 'caffe')):
        raise ValueError(f'The current model {model} is not the onnx '
                         'or caffe model, please specify the config file '
                         'and re-execute hb_compile')
    model_type = 'onnx'
    if proto:
        model_type = 'caffe'
    yaml_path = YamlBuilder(mode="check",
                            model=model,
                            proto=proto,
                            model_type=model_type,
                            march=march,
                            input_shape=input_shape).build()
    conf = params_check(yaml_path=yaml_path)
    ptq_model = ptq_model_build(conf)
    return conf, ptq_model


def hbm_build(conf: ConfigInfo,
              ptq_model: ModelProto,
              skip: str = None) -> None:
    hbm_builder = HBMBuilder(ptq_model=ptq_model,
                             march=conf.march,
                             config_info=conf,
                             skip=skip)
    hbm_builder.build()
    return None


def hbm_build_from_bc(model_path: str,
                      yaml_path: str,
                      march: str = '') -> None:
    if not model_path.endswith('.bc'):
        raise ValueError(f"model {model_path} is invalid. Only models with "
                         ".bc suffixes are supported")
    if not os.path.exists(model_path):
        raise ValueError(f"{model_path} does not exist.")

    model = load(model_path)
    sess = HBIRHandle(model)

    if not yaml_path:
        build_params = {}
        yaml_path = YamlBuilder(mode="fast_perf",
                                model=model_path,
                                proto='',
                                model_type="bc",
                                march=march,
                                input_shape=()).build()
        if sess.desc:
            build_params = sess.convert_model_info_to_dict()
            del build_params['model_parameters']['working_dir']
            if march:
                build_params['model_parameters']['march'] = march
        else:
            if not march:
                raise ValueError("Please specify the config yaml or march.")
            build_params = {"model_parameters": {"march": march}}
        update_yaml(yaml_path, build_params)
    else:
        if not os.path.exists(yaml_path):
            raise ValueError(f"{yaml_path} does not exist.")

    params_parser = ParamsParser(yaml_path=yaml_path)
    params_parser.validate_bc_compile_mode_parameters()
    conf = params_parser.conf

    hbm_builder = HBMBuilder(ptq_model=None,
                             march=conf.march,
                             config_info=conf)

    advice_path = conf.working_dir + "/"
    if sess.current_phase == 'export':
        sess.set_model_info(config_info=conf)
        hbm_builder.quantize_hbir = sess.convert_quantize_model(
            march=conf.march,
            advice=True,
            advice_path=advice_path,
            enable_vpu=conf.enable_vpu)
        hbm_builder.save(hbm_builder.quantize_hbir, "_quantized_model.bc")
    else:
        sess.update_desc_from_bc(config_info=conf)
        hbm_builder.quantize_hbir = model
    hbm_builder.remove_node()
    hbm_builder.compile_model()
    hbm_builder.hbm_perf()
    logging.info('The hb_compile completes running')
    copy_compile_log_file(conf.working_dir)
    return None


def fast_perf_mode(model: str,
                   proto: str,
                   march: str,
                   input_shape: Tuple[Tuple[str, str]],
                   skip: str = None):
    if not model.endswith(('.onnx', '.caffemodel')):
        raise ValueError("The --fast-perf parameter only supports"
                         "the .onnx and .caffemodel model.")
    conf, ptq_model = fast_perf_handle(model=model,
                                       proto=proto,
                                       march=march,
                                       input_shape=input_shape)
    hbm_build(conf, ptq_model, skip=skip)

    logging.info('The hb_compile completes running')
    copy_compile_log_file(conf.working_dir)


def onnx_config_mode(config: str, skip: str = None):
    conf = params_check(yaml_path=config)
    ptq_model = ptq_model_build(conf)
    hbm_build(conf, ptq_model, skip=skip)
    logging.info('The hb_compile completes running')
    copy_compile_log_file(conf.working_dir)


def bc_config_mode(config: str, model: str, march: str):
    if not model.endswith('.bc'):
        raise ValueError("Only recompiling the bc model is supported")
    hbm_build_from_bc(model, config, march=march)


def check_mode(model: str,
               proto: str,
               march: str,
               input_shape: Tuple[str, str],
               skip: str = None):
    conf, ptq_model = model_check_handle(model=model,
                                         proto=proto,
                                         march=march,
                                         input_shape=input_shape)
    hbm_build(conf, ptq_model, skip=skip)
    logging.info('The hb_compile completes running')
    copy_compile_log_file(conf.working_dir)


@click.command()
@click.help_option('--help', '-h')
@click.option('-c',
              '--config',
              type=click.Path(exists=True),
              help='Model convert config file')
@click.option('-m',
              '--model',
              type=click.Path(exists=True),
              help='Model to be compiled or modified')
@click.option('--proto',
              type=click.Path(exists=True),
              help='Caffe prototxt file')
@click.option('--march',
              type=click.Choice(march_list),
              help="BPU's micro architectures")
@click.option('-i',
              '--input-shape',
              type=(str, str),
              multiple=True,
              help='Specify the model input shape, '
              'e.g. --input-shape input1 1x3x224x224')
@click.option('--fast-perf',
              'fast_perf',
              flag_value=True,
              default=False,
              help='Build with fast perf mode, '
              'e.g. --fast-perf')
@click.option('--skip',
              default=None,
              type=str,
              help='Skip the specified steps,'
              'e.g. --skip compile')
@on_exception_exit
def main(config: str, model: str, proto: str, march: str, input_shape: tuple,
         fast_perf: bool, skip: str) -> None:
    """A tool that maps floating-point models to quantized models and provides some additional validation features
    """  # noqa
    log_level = logging.DEBUG
    init_root_logger('hb_compile', file_level=log_level)

    logging.info('Start hb_compile...')
    logging.info('hbdk version: %s', hbdk_version)
    logging.info('hmct version: %s', horizon_nn_version)
    logging.info('hb_compile version: %s', horizon_tc_ui_version)

    if fast_perf and config:
        logging.error(
            'fast-perf mode is turned on, '
            'the incoming config file %s cannot be used', config)
        raise ValueError('please consider turn off fast-perf or cancel the '
                         'incoming config file')
    if fast_perf and not model:
        raise ValueError("Please use -m/--model to provide model information")
    if not model and not config:
        raise ValueError('Please input either config or model')

    if config and not model and not fast_perf:  # config mode
        onnx_config_mode(config, skip=skip)
    elif fast_perf and not config and model:  # fast-perf mode
        fast_perf_mode(model, proto, march, input_shape, skip=skip)
    elif config and model and not fast_perf:
        if skip:
            logging.warning('The --skip option is not supported in this mode')
        bc_config_mode(config, model, march)
    elif model and not config and not fast_perf:
        check_mode(model, proto, march, input_shape, skip=skip)
    else:
        raise ValueError("Invalid parameter configuration")


if __name__ == '__main__':
    main()
