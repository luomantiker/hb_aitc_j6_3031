import re
from typing import Any, Dict, Optional, Sequence

from .quant_config import QuantConfig, tree


def load_quant_config_by_legacy(
    quant_config: QuantConfig,
    is_qat: Optional[bool] = False,
    cali_dict: Optional[Dict[str, Any]] = None,
    node_dict: Optional[Dict[str, Dict[str, Any]]] = None,
    optimization: Optional[Sequence[str]] = None,
):
    """加载之前版本的量化配置参数."""
    # load quant_config by cali_dict
    load_quant_config_by_calidict(quant_config, is_qat, cali_dict)
    # load quant_config by node_dict
    load_quant_config_by_nodedict(quant_config, node_dict)
    # load quant_config by optimization
    load_quant_config_by_opt(quant_config, optimization)


def load_quant_config_by_calidict(
    quant_config: QuantConfig,
    is_qat: bool,
    cali_dict: Optional[Dict[str, Any]] = None,
) -> None:
    """校准方法为"load",模型非QAT时报错."""
    if cali_dict is not None:
        act_type = cali_dict.get("calibration_type", "")
        if not is_qat and act_type == "load":
            raise ValueError(
                "ERROR: The model is not a QAT model but the calibration_type "
                "is load. Please modify calibration_type because 'load' is only "
                "used by QAT model."
            )

    """解析cali_dict, 并集成到quant_config中."""

    def collect_weight_config(cali_dict: Dict[str, Any]) -> Dict[str, Any]:
        """兼容cali_dict配置到weight_config."""
        weight_config = tree()
        # qat model
        if is_qat:
            weight_config["calibration_type"] = "load"
        # weight percentile
        if "weight_percentile" in cali_dict:
            weight_config["max_percentile"] = cali_dict.pop("weight_percentile")
        # bias correction
        if "bias_sample" in cali_dict:
            weight_config["bias_correction"].setdefault(
                "num_sample", cali_dict.pop("bias_sample")
            )
        if "bias_metric" in cali_dict:
            weight_config["bias_correction"].setdefault(
                "metric", cali_dict.pop("bias_metric")
            )
        return weight_config

    def collect_activation_config(cali_dict: Dict[str, Any]) -> Dict[str, Any]:
        """兼容cali_dict配置到activation_config."""
        activation_config = tree()
        # load calibration params for activation
        if "calibration_type" in cali_dict:
            activation_config["calibration_type"] = cali_dict["calibration_type"]
        if "num_bin" in cali_dict:
            activation_config["num_bin"] = cali_dict.pop("num_bin")
        if "max_num_bin" in cali_dict:
            activation_config["max_num_bin"] = cali_dict.pop("max_num_bin")
        if "max_percentile" in cali_dict:
            activation_config["max_percentile"] = cali_dict.pop("max_percentile")
        if "per_channel" in cali_dict:
            activation_config["per_channel"] = cali_dict.pop("per_channel")
        if "strategy" in cali_dict:
            activation_config["strategy"] = cali_dict.pop("strategy")
        # parse default calibration type
        if activation_config.get("calibration_type") == "default":
            activation_config["calibration_type"] = ["max", "kl"]
            activation_config["max_percentile"] = [0.99995, 1.0]
            activation_config["per_channel"] = [True, False]
            activation_config["asymmetric"] = [True, False]
        # qat model or empty calibration data
        if is_qat:
            activation_config["calibration_type"] = "load"
        elif not cali_dict.get("calibration_data") and not cali_dict.get(
            "calibration_loader"
        ):
            activation_config["calibration_type"] = "fixed"

        return activation_config

    def collect_modelwise_config(cali_dict: Dict[str, Any]) -> Dict[str, Any]:
        """兼容cali_dict配置到modelwise_search."""
        modelwise_search = tree()
        if cali_dict.get("calibration_type") == "default":
            modelwise_search["similarity"] = 0.995
        if "default_metric" in cali_dict:
            modelwise_search["metric"] = cali_dict.pop("default_metric")
        if "default_similarity" in cali_dict:
            modelwise_search["similarity"] = cali_dict.pop("default_similarity")
        return modelwise_search

    def collect_layerwise_config(cali_dict: Dict[str, Any]) -> Dict[str, Any]:
        """兼容cali_dict配置到layerwise_search."""
        layerwise_search = tree()
        if "mix_metric" in cali_dict:
            layerwise_search["metric"] = cali_dict.pop("mix_metric")
        if "mix_topk" in cali_dict:
            layerwise_search["topk"] = cali_dict.pop("mix_topk")
        if "use_int16" in cali_dict:
            qtype = "int16" if cali_dict.pop("use_int16") else "int8"
            layerwise_search["qtype"] = qtype
        return layerwise_search

    cali_dict = {} if cali_dict is None else cali_dict
    # convert cali_dict to quant_config
    weight_config = collect_weight_config(cali_dict)
    if weight_config:
        quant_config.model_config["weight"].update(weight_config)
    activation_config = collect_activation_config(cali_dict)
    if activation_config:
        quant_config.model_config["activation"].update(activation_config)
    modelwise_config = collect_modelwise_config(cali_dict)
    if modelwise_config:
        quant_config.model_config["modelwise_search"].update(modelwise_config)
    layerwise_config = collect_layerwise_config(cali_dict)
    if layerwise_config:
        quant_config.model_config["layerwise_search"].update(layerwise_config)
    quant_config.check_quant_config()


def load_quant_config_by_nodedict(
    quant_config: QuantConfig,
    node_dict: Optional[Dict[str, Dict[str, Any]]] = None,
):
    # 将node_dict中的配置集成到quant_config中
    if node_dict is not None:
        for node_name, node_config in node_dict.items():
            if "ON" in node_config:
                if node_config["ON"] == "BPU":
                    quant_config.node_config[node_name].update({"InputType": "int8"})
                elif node_config["ON"] == "CPU":
                    quant_config.node_config[node_name].update({"InputType": "float32"})
                del node_config["ON"]
            quant_config.node_config[node_name].update(node_config)
    quant_config.check_quant_config()


def load_quant_config_by_opt(
    quant_config: QuantConfig,
    optimization: Optional[Sequence[str]] = None,
):
    # 将optimization中量化相关的配置集成到quant_config, 当前量化相关配置包括
    # 1. set_all_nodes_{dtype}
    # 2. set_model_output_{dtype}
    # 3. set_{nodekind}_{input/output}_{dtype}
    # 4. asymmetric, bias_correction
    if optimization is not None:
        quant_config.optimization = list(optimization)
        for opt in optimization:
            remove_from_optimization = True
            split_str_vec = opt.split("_")
            if opt == "run_fast":
                quant_config.model_config["all_node_type"] = "int8"
                quant_config.model_config["model_output_type"] = "int8"
                remove_from_optimization = False
            elif opt.startswith("set_all_nodes_"):
                quant_config.model_config["all_node_type"] = split_str_vec[-1]
            elif opt.startswith("set_model_output_"):
                quant_config.model_config["model_output_type"] = split_str_vec[-1]
            elif re.match(r"^set_.*_input_.*$", opt):
                quant_config.op_config[split_str_vec[1]]["InputType"] = split_str_vec[
                    -1
                ]
            elif re.match(r"^set_.*_output_.*$", opt):
                quant_config.op_config[split_str_vec[1]]["OutputType"] = split_str_vec[
                    -1
                ]
            elif opt == "asymmetric":
                quant_config.model_config["activation"]["asymmetric"] = True
            elif opt == "bias_correction":
                bias_correction = quant_config.model_config["weight"]["bias_correction"]
                bias_correction["num_sample"] = bias_correction.get("num_sample", 1)
                bias_correction["metric"] = bias_correction.get(
                    "metric", "cosine-similarity"
                )
            else:
                remove_from_optimization = False
            if opt in quant_config.optimization and remove_from_optimization:
                quant_config.optimization.remove(opt)
            elif opt not in quant_config.optimization and not remove_from_optimization:
                quant_config.optimization.append(opt)

    quant_config.check_quant_config()
