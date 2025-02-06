# Copyright (c) 2021 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import logging

import click
from horizon_tc_ui.eval_preprocess import __VERSION__, EvalPreprocess
from horizon_tc_ui.eval_preprocess.conf import MODEL_DICT
from horizon_tc_ui.utils.tool_utils import init_root_logger, on_exception_exit


@click.command()
@click.help_option('--help', '-h')
@click.version_option(version=__VERSION__)
@click.option('--model_name',
              '-m',
              type=click.Choice([k for k, v in MODEL_DICT.items() if v['enable'] is True]),
              required=True,
              help='Input model name.')
@click.option('--image_dir',
              '-i',
              type=str,
              required=True,
              help='Input image dir.')
@click.option('--output_dir',
              '-o',
              type=str,
              default='affected',
              help='Output dir.')
@click.option('--val_txt', '-v', type=str, default=None, hidden=False)
@on_exception_exit
def cmd_main(image_dir, model_name, output_dir, val_txt):
    """
    Example: hb_eval_preprocess -m mobilenetv1 -i ./files
    """
    init_root_logger("hb_eval_preprocess", logging.INFO)
    eval_preprocess = EvalPreprocess(image_dir=image_dir,
                                     model_name=model_name,
                                     output_dir=output_dir,
                                     val_txt_path=val_txt)
    eval_preprocess.run()
