# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import logging

import click

from horizon_tc_ui.utils.tool_utils import init_root_logger, on_exception_exit
from horizon_tc_ui.verifier import VerifierParams
from horizon_tc_ui.verifier.comparator import VerifierComparator
from horizon_tc_ui.verifier.data_preprocess import VerifierDataPreprocess
from horizon_tc_ui.verifier.inference import VerifierInference
from horizon_tc_ui.verifier.params_check import VerifierParamsCheck
from horizon_tc_ui.version import __version__


def verifier(model,
             input_files: list,
             skip_sim: bool = False,
             skip_arm: bool = False,
             digits: int = 5,
             board_ip: str = "",
             username: str = "root",
             password: str = "",
             printable: bool = False) -> VerifierParams:
    checker = VerifierParamsCheck(models_path=model,
                                  input_files=input_files,
                                  skip_sim=skip_sim,
                                  skip_arm=skip_arm,
                                  digits=digits,
                                  board_ip=board_ip,
                                  username=username,
                                  password=password)
    params_info = checker.check()

    data_preprocess = VerifierDataPreprocess(params_info)
    data_preprocess.run()

    runner = VerifierInference(params_info)
    runner.run()

    comparator = VerifierComparator(params_info)
    comparator.run()
    if printable:
        comparator.print()
    return params_info


@click.command()
@click.help_option('--help', '-h')
@click.version_option(version=__version__)
@click.option('-m',
              '--model',
              type=str,
              required=True,
              help='The types of parameters supported include onnx/bc/'
              'hbm models, with multiple models separated by ",".')
@click.option('-b',
              '--board-ip',
              type=str,
              required=False,
              help='',
              hidden=True)
@click.option('-i',
              '--input',
              type=str,
              required=False,
              multiple=True,
              help='Original model input files')
@click.option('-s',
              '--run-sim',
              is_flag=True,
              default=None,
              help="Run simulation",
              hidden=True)
@click.option('--skip-arm',
              is_flag=True,
              default=False,
              help="Skip arm",
              hidden=True)
@click.option('--skip-sim',
              is_flag=True,
              default=False,
              help="Skip simulation",
              hidden=True)
@click.option('-r',
              '--dump-all-nodes-results',
              is_flag=True,
              default=False,
              help="Dump all nodes results",
              hidden=True)
@click.option('-c',
              '--compare_digits',
              type=int,
              default=5,
              required=False,
              help='The number of decimal places to compare')
@click.option('-u',
              '--username',
              type=str,
              default='root',
              help='Board username',
              hidden=True)
@click.option('-p',
              '--password',
              type=str,
              default='',
              help='Board password',
              hidden=True)
@on_exception_exit
def cmd_main(model: str, board_ip: str, input: tuple, run_sim: bool,
             skip_arm: bool, skip_sim: bool, compare_digits: int,
             username: str, password: str,
             dump_all_nodes_results: bool) -> None:
    log_level = logging.DEBUG
    init_root_logger('hb_verifier', file_level=log_level)

    logging.info("HB_Verifier Starts...")
    logging.info("verifier tool version %s", __version__)

    if run_sim and run_sim == skip_sim:
        raise ValueError(
            "Configuring both the --skip-sim and --run-sim parameters is not supported."  # noqa
        )
    if run_sim and run_sim != skip_sim:
        logging.error(
            "The --run-sim parameter is deprecated. Please use the --skip-sim parameter later."  # noqa
        )
        run_sim = not skip_sim
    if dump_all_nodes_results:
        logging.error(
            "The --dump-all-nodes-results parameter is not currently supported"
        )  # noqa
    if skip_sim and skip_arm:
        logging.error(
            "Configuring both --skip-arm and --skip-sim is not supported."
        )  # noqa

    verifier(model=model,
             input_files=list(input),
             skip_sim=skip_sim,
             skip_arm=skip_sim,
             digits=compare_digits,
             board_ip=board_ip,
             username=username,
             password=password,
             printable=True)


if __name__ == "__main__":
    cmd_main()
