# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import json
import logging
import os
from typing import Any

import click
from hbdk4.compiler import load
from hbdk4.compiler.utils.remove_io_op import get_removable_io_op

import horizon_tc_ui.version as hb_mapper_version
from horizon_tc_ui import HB_HBIRRuntime, HB_ONNXRuntime
from horizon_tc_ui.config.mapper_consts import removal_list
from horizon_tc_ui.hb_hbmruntime import HB_HBMRuntime
from horizon_tc_ui.hbm_handle import HBMHandle
from horizon_tc_ui.utils import model_utils, tool_utils
from horizon_tc_ui.utils.tool_utils import get_list_from_txt
from horizon_tc_ui.visualize import Visualize


@click.command(
    help="A tool to get info about model compilation parameters, properties")
@click.help_option('--help', '-h')
@click.version_option(version=hb_mapper_version.__version__)
@tool_utils.on_exception_exit
@click.option('-n',
              '--name',
              type=str,
              default="",
              help='Only output model compile info of desired model')
@click.option('-v',
              '--visualize',
              'visualize',
              flag_value=True,
              default=False,
              help='Start netron server to show model structure')
@click.argument('model_path', type=str)
def cmd_main(model_path: str, name: str, visualize: bool) -> None:
    """A Tool used to get the deps info, compile info or removable node info

    Example: hb_model_info foo.bin/foo.hbm/foo.bc

    Args:
        model_path (_type_): Input model path
        m (str): Deprecated
        name (str): Only output model compile info of desired model

    Raises:
        ValueError: If model type is not support
    """
    tool_utils.init_root_logger("hb_model_info")
    main(model_path, name, visualize)


def main(model_path: str, name: str, visualize: bool) -> None:
    """Triage to different processing logic based on model type

    Args:
        model_path (str): Input model path
        name (str): Only output model compile info of desired model

    Raises:
        ValueError: If model suffix not allow
        ValueError: If model doesn't exist
    """
    model_suffix = ('.hbm', '.bc', '.onnx')
    if not model_path.endswith(model_suffix):
        raise ValueError(
            f"model {model_path} is invalid. Only models with .hbm, .bc"
            " and .onnx suffixes are supported")
    desired_model = ''
    if name:
        desired_model = name if not name.endswith(model_suffix) \
            else name.split(".")[0]
    if not os.path.exists(model_path):
        raise ValueError(f"{model_path} does not exist !!!")

    if visualize:
        if not os.path.exists('.hb_model_info'):
            os.mkdir('.hb_model_info')

    logging.info("Start hb_model_info....")
    logging.info("hb_model_info version %s" % hb_mapper_version.__version__)
    if model_path.endswith('.hbm'):
        hbm_model_info(model_path, desired_model, visualize)
    elif model_path.endswith('.bc'):
        bc_model_info(model_path, visualize)
    elif model_path.endswith('.onnx'):
        onnx_model_info(model_path, visualize)


def bc_model_info(model_path: str, visualize: bool) -> None:
    sess = HB_HBIRRuntime(model_file=model_path)

    if sess.function.desc and tool_utils.validate_json_str(sess.function.desc):
        tc_desc = json.loads(sess.function.desc)
        logging.debug("bc model desc: %s", tc_desc)
        deps_log_info(tc_desc)
        model_parameters_log_info(tc_desc)
        input_parameters_log_info(tc_desc)
        calibration_parameters_log_info(tc_desc)
        if tc_desc.get('CUSTOM_OP_METHOD'):
            custom_op_log_info(tc_desc)
        misc_log_info(tc_desc)
        bc_removable_node_info(model_path)

    print_input_and_output_info(sess)

    if visualize:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        onnx_save_path = f'.hb_model_info/{model_name}.onnx'
        Visualize(model_path=model_path, save_path=onnx_save_path).visualize()


def hbm_model_info(model_path: str,
                   desired_model: str = '',
                   visualize: bool = False) -> None:
    hbm_handle = HBMHandle(model_path)
    hbm_desc = hbm_handle.desc()
    if hbm_desc:
        model_names = list(hbm_desc['models'].keys())
        logging.debug(hbm_desc)

        if desired_model:
            if desired_model not in model_names:
                raise ValueError("Desired model invalid: "
                                 f"{desired_model} not in {model_names}")
            model_names = [desired_model]
        else:
            if len(model_names) > 1:
                logging.warning("This model is a packaged model, "
                                "unspecified models will default "
                                "to output all model info")
        for model_name in model_names:
            logging.info("************* %s *************", model_name)
            tc_desc_str = hbm_desc['models'][model_name]['desc']

            if tc_desc_str and tool_utils.validate_json_str(tc_desc_str):
                tc_desc = json.loads(tc_desc_str)
                logging.debug("hbm %s desc: %s", model_name, tc_desc)
                deps_log_info(tc_desc)
                model_parameters_log_info(tc_desc)
                input_parameters_log_info(tc_desc)
                calibration_parameters_log_info(tc_desc)
                if tc_desc.get('CUSTOM_OP_METHOD'):
                    custom_op_log_info(tc_desc)
                hbm_compiler_parameters_log_info(tc_desc)
                misc_log_info(tc_desc)
        if len(model_names) <= 1:
            sess = HB_HBMRuntime(model_path)
            print_input_and_output_info(sess=sess)

    if visualize:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        onnx_save_path = f'.hb_model_info/{model_name}.prototxt'
        Visualize(model_path=model_path, save_path=onnx_save_path).visualize()


def onnx_model_info(model_path: str, visualize: bool) -> None:
    sess = HB_ONNXRuntime(model_file=model_path)
    if len(sess.model.opset_import) > 0:
        logging.info("opset version: %s", sess.model.opset_import[0].version)
    print_input_and_output_info(sess=sess)
    if visualize:
        Visualize(model_path=model_path, save_path=model_path).visualize()


def print_input_and_output_info(sess: Any) -> None:
    model_info = [('NAME', 'TYPE', 'SHAPE', 'DATA_TYPE')]
    mapping = {'input': sess.input_names, 'output': sess.output_names}
    for prefix, op_names in mapping.items():
        for idx, op_name in enumerate(op_names):
            shape = getattr(sess, f'{prefix}_shapes')[idx]
            data_type = getattr(sess, f'{prefix}_types')[idx]
            if isinstance(data_type, int):
                data_type = model_utils.DataType(data_type).name
            elif hasattr(data_type, "__name__"):
                data_type = data_type.__name__.upper()
            else:
                data_type = str(data_type).upper()
            model_info.append((op_name, prefix, shape, data_type))
    logging.info("############# Model input/output info #############")
    tool_utils.tabulate(model_info)


def deps_log_info(model_info) -> None:
    logging.info("############# model deps info #############")
    params = ["BUILDER_VERSION", "HBDK_VERSION", "HORIZON_NN_VERSION"]
    for param in params:
        param_padding = "{:<20}".format(param.lower().replace("_", " "))
        logging.info(f'{param_padding}: {model_info[param]}')
    return None


def model_parameters_log_info(model_info) -> None:
    logging.info("############# model_parameters info #############")
    params = [
        'CAFFE_MODEL', 'PROTOTXT', 'ONNX_MODEL', 'MARCH', 'LAYER_OUT_DUMP',
        'LOG_LEVEL', 'WORKING_DIR', 'MODEL_PREFIX', 'OUTPUT_LAYOUT',
        'OUTPUT_NODES', 'REMOVE_NODE_TYPE', 'REMOVE_NODE_NAME',
        'SET_NODE_DATA_TYPE', 'DEBUG_MODE', 'NODE_INFO'
    ]
    for param in params:
        if param not in model_info or not model_info[param]:
            continue
        params_map = {
            'MARCH': 'BPU march',
            'WORKING_DIR': 'working dir',
            'MODEL_PREFIX': 'output_model_file_prefix'
        }
        param_padding = "{:<20}".format(param.lower())
        if param in params_map:
            param_padding = "{:<20}".format(params_map[param])
        logging.info(f'{param_padding}: {model_info[param]}')
    return None


def bc_removable_node_info(bc_model: str) -> None:
    """Print bc model removable node info

    Args:
        bc_model (str): bc model path
    """
    try:
        load_model = load(bc_model)
    except Exception as error:
        raise ValueError(f"Model {bc_model} load failed, reason: {error}")

    removable_info = get_removable_io_op(func=load_model[0],
                                         op_types=removal_list)

    if not removable_info:
        logging.info(f"The model {bc_model} has no nodes to delete")
        return None
    removable_info.insert(0, ("Node Name", "Node Type"))
    logging.info("############# Removable node info #############")
    tool_utils.tabulate(removable_info)


def input_parameters_log_info(model_info) -> None:
    logging.info("############# input_parameters info #############")
    params_list = [
        'INPUT_NAMES', 'INPUT_TYPE_RT', 'INPUT_SPACE_AND_RANGE',
        'INPUT_TYPE_TRAIN', 'INPUT_LAYOUT_RT', 'INPUT_LAYOUT_TRAIN',
        'NORM_TYPE', 'INPUT_SHAPE', 'MEAN_VALUE', 'SCALE_VALUE', 'STD_VALUE',
        'INPUT_BATCH', 'SEPARATE_BATCH', 'SEPARATE_NAME'
    ]
    params_list_map = {}
    for param in params_list:
        if param not in model_info:
            continue
        params_list_map[param] = get_list_from_txt(model_info[param])

        params_map = {
            'INPUT_NAMES': 'input_name',
            'INPUT_SPACE_AND_RANGE': 'input_space&range',
            'CALI_DIR': 'cal_data_dir',
            'CAL_DATA_DIR': 'cal_data_dir'
        }
    logging.info("------------------------------------------")
    for ind, name in enumerate(params_list_map['INPUT_NAMES']):
        logging.info(f"---------input info : {name} ---------")
        for param, values in params_list_map.items():
            if len(values) == 1:
                ind = 0
            if values and values[ind]:
                param_padding = "{:<20}".format(param.lower())
                if param in params_map:
                    param_padding = "{:<20}".format(params_map[param])
                logging.info(f'{param_padding}: {values[ind]}')
        logging.info(f"---------input info : {name} end -------")
    logging.info("------------------------------------------")
    return None


def calibration_parameters_log_info(model_info) -> None:
    logging.info("############# calibration_parameters info #############")
    params = [
        'PREPROCESS_ON', 'CALI_TYPE', 'CALI_DIR', 'MAX_PERCENTILE',
        'OPTIMIZATION', 'PER_CHANNEL', 'RUN_ON_CPU', 'RUN_ON_BPU',
        '16BIT_QUANTIZE', 'CAL_DATA_DIR', 'QUANT_CONFIG'
    ]
    for param in params:
        if param not in model_info or not model_info[param]:
            continue
        params_map = {
            'CALI_TYPE': 'calibration_type',
            '16BIT_QUANTIZE': 'working dir',
            'MODEL_PREFIX': '16 bit quantize',
            'CALI_DIR': 'cal_data_dir'
        }
        param_padding = "{:<20}".format(param.lower())
        if param in params_map:
            param_padding = "{:<20}".format(params_map[param])
        logging.info(f'{param_padding}: {model_info[param]}')
    if 'CALI_EXTRA_PARAM' in model_info:
        for key, value in model_info['CALI_EXTRA_PARAM'].items():
            logging.info(f'{"{:<20}".format(key)}: {value}')
    return None


def hbm_compiler_parameters_log_info(model_info) -> None:
    logging.info("############# compiler_parameters info #############")
    params = [
        'DEBUG', 'OPTIMIZE_LEVEL', 'COMPILE_MODE', 'CORE_NUM',
        'MAX_TIME_PER_FC', 'ADVICE', 'BALANCE_FACTOR', 'ABILITY_ENTRY',
        'INPUT_SOURCE', 'OPTIMIZATION_LEVEL', 'hbdk3_compatible_mode',
        'EXTRA_PARAMS'
    ]
    for param in params:
        if param not in model_info or not model_info[param]:
            continue
        params_map = {
            'OPTIMIZATION_LEVEL': 'optimize_level',
        }
        param_padding = "{:<20}".format(param.lower())
        if param in params_map:
            param_padding = "{:<20}".format(params_map[param])
        logging.info(f'{param_padding}: {model_info[param]}')
    return None


def custom_op_log_info(model_info) -> None:
    logging.info("############# custom_op info #############")
    params = [
        'CUSTOM_OP_METHOD',
        'CUSTOM_OP_DIR',
        'CUSTOM_OP_REGISTER_FILES',
    ]
    for param in params:
        if param not in model_info or not model_info[param]:
            continue
        # TODO(wei04.wei): The key of model_info to be modified
        params_map = {'CUSTOM_OP_REGISTER_FILES': 'custom_op_reg_files'}
        param_padding = "{:<20}".format(param.lower())
        if param in params_map:
            param_padding = "{:<20}".format(params_map[param])
        logging.info(f'{param_padding}: {model_info[param]}')
    return None


def misc_log_info(model_info) -> None:
    deleted_nodes = model_info.get("DELETED_NODES", "")
    if deleted_nodes != "":
        logging.warning(
            "Please note that the model information shown now is the "
            "information when the model was compiled, this model has been "
            "modified.")
        logging.info("--------- deleted nodes -------------------")
        with open("deleted_nodes_info.txt", "w") as eval_log_handle:
            for item in deleted_nodes.split():
                logging.info(f'deleted nodes: {item}')
                deleted_node_info = model_info["NODE_" + item].replace(
                    ',', '').replace(']', '').replace('[', '')
                eval_log_handle.write(f"{deleted_node_info}\n")
    runtime_info = model_info.get("RUNTIME_INFO", "")
    if runtime_info:
        logging.debug("--------- runtime_info -------------------")
        for item in runtime_info.split():
            logging.debug(f'runtime_info {item}: {model_info[item]}')
    return None


if __name__ == "__main__":
    cmd_main()
