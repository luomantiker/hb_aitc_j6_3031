import ast
import functools
import logging
import operator
import os
import re

import click

from hmct.ir import load_model
import hmct.quantizer.debugger as dbg
from hmct.quantizer.debugger.node_info import NodeInfo
from hmct.quantizer.debugger.save_node_sensitivity import save_node_sensitivity
from hmct.quantizer.debugger.util import check_parameters


def keep_list_format(string):
    """转换列表形式字符串为列表.

    支持的string形式包括: 1. 字符串单列表([xxx,xxx],['xxx','xxx'],'["xxx","xxx"]',
    "['xxx','xxx']"); 2. 字符串嵌套列表([[xxx,xxx],[xxx]],[['xxx','xxx'],['xxx']],
    '[["xxx","xxx"],["xxx"]]',"[['xxx','xxx'],['xxx']]").
    """
    if string is None:
        return None
    if string[0] != "[":
        return string
    string = re.sub(r"['\"]", "", string)
    string = re.sub(r"[^\[\],\s]+", r"'\g<0>'", string)
    return ast.literal_eval(string)


CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@click.group(context_settings=CONTEXT_SETTINGS)
def debugger():
    pass


@debugger.command()
# Required parameters
@click.argument("model_or_file")
@click.argument("calibrated_data")
# save path
@click.option(
    "-s",
    "--save_dir",
    default="./debug_result/",
    help="The path used to save result.",
    required=False,
)
# node sensitivity parameters
@click.option(
    "-nm",
    "--ns_metrics",
    default="cosine-similarity",
    help="The method to caculate node sensitivity.",
    required=False,
)
@click.option(
    "-o",
    "--output_node",
    default=None,
    help="Select which node's output is used to calculate node sensitivity.",
    required=False,
)
@click.option(
    "-nt",
    "--node_type",
    default=None,
    help="Type of node for calculating sensitivity.",
    required=False,
)
@click.option(
    "-dn",
    "--data_num",
    default=1,
    type=int,
    help="The amount of data required to calculate node sensitivity.",
    required=False,
)
@click.option(
    "-v",
    "--verbose",
    default=False,
    type=bool,
    help="Display node sensitivity information on the terminal if True, else False.",
    required=False,
)
@click.option(
    "-i",
    "--interested_nodes",
    default=None,
    help="Nodes involved in sensitivity sorting.By default, all nodes "
    "participate in sorting.",
    required=False,
)
# distribution parameters
@click.option(
    "-dnl",
    "--dis_nodes_list",
    default=None,
    help="Nodes to print distribution.",
    required=False,
)
# channelwise data distribution parameters
@click.option(
    "-cn",
    "--cw_nodes_list",
    default=None,
    help="Nodes to print channelwise data distribution.",
    required=False,
)
@click.option(
    "-a",
    "--axis",
    default=None,
    type=int,
    help="Index of node input data channel.",
    required=False,
)
# accumulate error parameters
@click.option(
    "-qn",
    "--quantize_node",
    default=None,
    help="Set nodes to be quantified.",
    required=False,
)
@click.option(
    "-nqn",
    "--non_quantize_node",
    default=None,
    help="Set nodes to be unquantified.",
    required=False,
)
@click.option(
    "-am",
    "--ae_metric",
    default="cosine-similarity",
    help="Calculation method of accumulation error.",
    required=False,
)
@click.option(
    "-avm",
    "--average_mode",
    default=False,
    type=bool,
    help="If it is set to true, the function will get the average of "
    "accumulation error.",
    required=False,
)
@click.option(
    "-pt",
    "--pick_threshold",
    default=0.999,
    type=float,
    help="Threshold for selecting sensitive nodes",
    required=False,
)
@click.option(
    "-sn",
    "--sensitive_nodes",
    default=None,
    help="Sensitive nodes",
    required=False,
)
def runall(
    model_or_file,
    calibrated_data,
    save_dir,
    ns_metrics,
    output_node,
    node_type,
    data_num,
    verbose,
    interested_nodes,
    dis_nodes_list,
    cw_nodes_list,
    axis,
    quantize_node,
    non_quantize_node,
    ae_metric,
    average_mode,
    pick_threshold,
    sensitive_nodes,
):
    logging.getLogger().setLevel(logging.INFO)
    if sensitive_nodes is None:
        sensitive_nodes = []
    check_parameters(
        model_or_file=model_or_file,
        calibrated_data=calibrated_data,
        data_num=data_num,
        metrics=keep_list_format(ns_metrics),
        nodes_list=keep_list_format(dis_nodes_list),
    )

    def flatten(nested_list):
        return (
            functools.reduce(operator.iadd, map(flatten, nested_list), [])
            if isinstance(nested_list, list)
            else [nested_list]
        )

    if non_quantize_node is not None:
        flatten_non_quantize_node = flatten(non_quantize_node)
        if (
            "weight" in flatten_non_quantize_node
            or "activation" in flatten_non_quantize_node
        ):
            raise ValueError(
                "In acc_error section, cannot set 'non_quantize_node' "
                "to 'weight' or 'activation'.",
            )

    check_parameters(metrics=keep_list_format(ae_metric))

    # node sensitivity
    nodes_list = []
    ns_save_dir = save_dir + "node_sensitivity/"
    if not os.path.exists(ns_save_dir):
        os.makedirs(ns_save_dir)
    if node_type is not None:
        node_type = keep_list_format(node_type)
    else:
        node_type = ["weight", "activation"]
    data_num = int(data_num) if data_num is not None else None
    if isinstance(node_type, list):
        for n in node_type:
            if n[0] == "'":
                n = n[1:-1]
            node_message = dbg.get_sensitivity_of_nodes(
                model_or_file=model_or_file,
                metrics=keep_list_format(ns_metrics),
                calibrated_data=calibrated_data,
                output_node=keep_list_format(output_node),
                node_type=n,
                data_num=data_num,
                verbose=verbose,
                interested_nodes=keep_list_format(interested_nodes),
            )
            if len(node_message.keys()) >= 5:
                nodes_list.extend(list(node_message.keys())[:5])
            else:
                nodes_list.extend(list(node_message.keys()))
            save_path = ns_save_dir + "sensitivity_of_" + n + ".log"
            save_node_sensitivity(node_message, save_path, n)
    else:
        node_message = dbg.get_sensitivity_of_nodes(
            model_or_file=model_or_file,
            metrics=keep_list_format(ns_metrics),
            calibrated_data=calibrated_data,
            output_node=keep_list_format(output_node),
            node_type=node_type,
            data_num=data_num,
            verbose=verbose,
            interested_nodes=keep_list_format(interested_nodes),
        )
        if len(node_message.keys()) >= 5:
            nodes_name = list(node_message.keys())[:5]
        else:
            nodes_name = list(node_message.keys())
        if node_type == "node":
            model = load_model(model_or_file).proto
            node_info = NodeInfo(model)
            for node_name in nodes_name:
                nodes_list.extend(node_info.get_calibration_by_node(node_name))
        else:
            nodes_list = nodes_name
        nodes_list = list(set(nodes_list))
        save_path = ns_save_dir + "sensitivity_of_" + node_type + ".log"
        save_node_sensitivity(node_message, save_path, node_type)
    # data distribution
    dd_save_dir = save_dir + "data_distribution/"
    if not os.path.exists(dd_save_dir):
        os.makedirs(dd_save_dir)
    if dis_nodes_list is None:
        dis_nodes_list = nodes_list
    dbg.plot_distribution(
        save_dir=dd_save_dir,
        model_or_file=model_or_file,
        calibrated_data=calibrated_data,
        nodes_list=dis_nodes_list,
    )
    # channelwise data distribution
    cd_save_dir = save_dir + "channelwise_distribution/"
    if not os.path.exists(cd_save_dir):
        os.makedirs(cd_save_dir)
    if cw_nodes_list is None:
        cw_nodes_list = nodes_list
    else:
        cw_nodes_list = keep_list_format(cw_nodes_list)
    dbg.get_channelwise_data_distribution(
        save_dir=cd_save_dir,
        model_or_file=model_or_file,
        calibrated_data=calibrated_data,
        nodes_list=keep_list_format(cw_nodes_list),
        axis=axis,
    )
    # accumulate error
    ae_save_dir = save_dir + "accumulate_error/"
    if not os.path.exists(ae_save_dir):
        os.makedirs(ae_save_dir)
    if quantize_node is None and non_quantize_node is None:
        quantize_node = ["weight", "activation"]
    else:
        quantize_node = keep_list_format(quantize_node)
        non_quantize_node = keep_list_format(non_quantize_node)
    dbg.plot_acc_error(
        save_dir=ae_save_dir,
        calibrated_data=calibrated_data,
        model_or_file=model_or_file,
        quantize_node=keep_list_format(quantize_node),
        non_quantize_node=keep_list_format(non_quantize_node),
        metric=ae_metric,
        average_mode=average_mode,
    )
    # sensitivity analysis
    sa_save_dir = save_dir + "sensitivity_analysis/"
    if not os.path.exists(sa_save_dir):
        os.makedirs(sa_save_dir)
    if not isinstance(sensitive_nodes, list):
        sensitive_nodes = keep_list_format(sensitive_nodes)
    if (
        node_type == "node"
        and "cosine-similarity" in ns_metrics
        and len(sensitive_nodes) == 0
    ):
        for k, v in node_message.items():
            if v["cosine-similarity"] <= pick_threshold:
                sensitive_nodes.append(k)
        if len(sensitive_nodes) == 0:
            logging.info(
                "pick_threshold is too small, there is no sensitive node "
                "that meets the requirements, please increase pick_threshold.",
            )
    dbg.sensitivity_analysis(
        model_or_file,
        calibrated_data,
        pick_threshold,
        data_num,
        sensitive_nodes,
        sa_save_dir,
    )


@debugger.command()
@click.argument("model_or_file")
@click.option(
    "-m",
    "--metrics",
    default="cosine-similarity",
    help="The method to caculate node sensitivity.",
    required=False,
)
@click.argument("calibrated_data")
@click.option(
    "-o",
    "--output_node",
    default=None,
    help="Select which node's output is used to calculate node sensitivity.",
    required=False,
)
@click.option(
    "-n",
    "--node_type",
    default="node",
    type=click.Choice(["node", "weight", "activation"]),
    help="Type of node for calculating sensitivity.",
    required=False,
)
@click.option(
    "-d",
    "--data_num",
    default=1,
    type=int,
    help="The amount of data required to calculate node sensitivity.",
    required=False,
)
@click.option(
    "-v",
    "--verbose",
    default=False,
    type=bool,
    help="Display node sensitivity information on the terminal if True, else False.",
    required=False,
)
@click.option(
    "-i",
    "--interested_nodes",
    default=None,
    help="Nodes involved in sensitivity sorting. By default, all nodes "
    "participate in sorting.",
    required=False,
)
def get_sensitivity_of_nodes(
    model_or_file,
    metrics,
    calibrated_data,
    output_node,
    node_type,
    data_num,
    verbose,
    interested_nodes,
):
    logging.getLogger().setLevel(logging.INFO)
    metrics = keep_list_format(metrics)
    output_node = keep_list_format(output_node)
    interested_nodes = keep_list_format(interested_nodes)
    data_num = int(data_num) if data_num is not None else None
    dbg.get_sensitivity_of_nodes(
        model_or_file=model_or_file,
        metrics=metrics,
        calibrated_data=calibrated_data,
        output_node=output_node,
        node_type=node_type,
        data_num=data_num,
        verbose=verbose,
        interested_nodes=interested_nodes,
    )


@debugger.command()
@click.option(
    "-s",
    "--save_dir",
    default="./debug_result/",
    help="The path used to save result.",
    required=False,
)
@click.argument("model_or_file")
@click.argument("calibrated_data")
@click.option(
    "-n",
    "--nodes_list",
    prompt="Specify nodes",
    help="Nodes to print distribution.",
    required=True,
)
def plot_distribution(save_dir, model_or_file, calibrated_data, nodes_list):
    nodes_list = keep_list_format(nodes_list)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    dbg.plot_distribution(
        save_dir=save_dir,
        model_or_file=model_or_file,
        calibrated_data=calibrated_data,
        nodes_list=nodes_list,
    )


@debugger.command()
@click.option(
    "-s",
    "--save_dir",
    default="./debug_result/",
    help="The path used to save result.",
    required=False,
)
@click.argument("model_or_file")
@click.argument("calibrated_data")
@click.option(
    "-n",
    "--nodes_list",
    prompt="Specify nodes",
    help="Nodes to print distribution.",
    required=True,
)
@click.option(
    "-a",
    "--axis",
    default=None,
    type=int,
    help="Index of node input data channel.",
    required=False,
)
def get_channelwise_data_distribution(
    save_dir,
    model_or_file,
    calibrated_data,
    nodes_list,
    axis,
):
    nodes_list = keep_list_format(nodes_list)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    dbg.get_channelwise_data_distribution(
        save_dir=save_dir,
        model_or_file=model_or_file,
        calibrated_data=calibrated_data,
        nodes_list=nodes_list,
        axis=axis,
    )


@debugger.command()
@click.option(
    "-s",
    "--save_dir",
    default="./debug_result/",
    help="The path used to save result.",
    required=False,
)
@click.argument("model_or_file")
@click.argument("calibrated_data")
@click.option(
    "-q",
    "--quantize_node",
    default=None,
    help="Set nodes to be quantified.",
    required=False,
)
@click.option(
    "-nq",
    "--non_quantize_node",
    default=None,
    help="Set nodes to be unquantified.",
    required=False,
)
@click.option(
    "-m",
    "--metric",
    default="cosine-similarity",
    help="Calculation method of accumulation error.",
    required=False,
)
@click.option(
    "-a",
    "--average_mode",
    default=False,
    type=bool,
    help="If it is set to true, the function will get the average of "
    "accumulation error.",
    required=False,
)
def plot_acc_error(
    save_dir,
    calibrated_data,
    model_or_file,
    quantize_node,
    non_quantize_node,
    metric,
    average_mode,
):
    quantize_node = (
        keep_list_format(quantize_node) if quantize_node is not None else None
    )
    non_quantize_node = (
        keep_list_format(non_quantize_node) if non_quantize_node is not None else None
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    dbg.plot_acc_error(
        save_dir=save_dir,
        calibrated_data=calibrated_data,
        model_or_file=model_or_file,
        quantize_node=quantize_node,
        non_quantize_node=non_quantize_node,
        metric=metric,
        average_mode=average_mode,
    )


@debugger.command()
@click.argument("model_or_file")
@click.argument("calibrated_data")
@click.option(
    "-p",
    "--pick_threshold",
    default=0.999,
    type=float,
    help="Threshold for selecting sensitive nodes",
    required=False,
)
@click.option(
    "-d",
    "--data_num",
    default=1,
    type=int,
    help="The amount of data required to calculate node sensitivity.",
    required=False,
)
@click.option(
    "-sn",
    "--sensitive_nodes",
    default=None,
    help="Sensitive nodes",
    required=False,
)
@click.option(
    "-sd",
    "--save_dir",
    default="./debug_result/",
    help="Save path",
    required=False,
)
def sensitivity_analysis(
    model_or_file,
    calibrated_data,
    pick_threshold,
    data_num,
    sensitive_nodes,
    save_dir,
):
    if sensitive_nodes is None:
        sensitive_nodes = []
    if not isinstance(sensitive_nodes, list):
        sensitive_nodes = keep_list_format(sensitive_nodes)
    dbg.sensitivity_analysis(
        model_or_file,
        calibrated_data,
        pick_threshold,
        data_num,
        sensitive_nodes,
        save_dir,
    )
