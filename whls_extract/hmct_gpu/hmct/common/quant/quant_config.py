from collections import defaultdict
import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Type, Union

from hmct.ir import OnnxModel, OnnxNode, extract_submodel


def tree():
    return defaultdict(tree)


def check_type(
    value: Union[Any, List[Any]],
    item_type: Optional[Type] = None,
    list_type: Optional[Type] = None,
) -> bool:
    """检查value的类型是否符合预期."""
    if item_type and isinstance(value, item_type):
        return True
    if (
        list_type
        and isinstance(value, list)
        and all(isinstance(v, list_type) for v in value)
    ):
        return True
    expect_types = []
    if item_type:
        expect_types.append(f"{item_type.__name__}")
    if list_type:
        expect_types.append(f"list({list_type.__name__})")
    expect_types = " or ".join(expect_types)
    raise TypeError(
        f"type(value) should be {expect_types}, but got {type(value).__name__}!"
    )


def remove_duplicates(config: Union[Any, List[Any]]) -> Union[Any, List[Any]]:
    """保持原有顺序去重."""
    if not isinstance(config, list):
        return config
    return list(dict.fromkeys(config))


class QuantConfig:
    def __init__(
        self,
        march: Optional[str] = None,
    ):
        self._march = march
        self._optimization = []
        # init quant_config, a dict that will contain all quantization configs
        self._quant_config = tree()

    @property
    def march(self) -> Union[str, None]:
        return self._march

    @property
    def optimization(self) -> List[str]:
        return self._optimization

    @optimization.setter
    def optimization(self, optimization: Sequence[str]) -> List[str]:
        self._optimization = optimization

    @property
    def activation_config(self) -> Union[Dict[str, Any], None]:
        return self._quant_config["model_config"]["activation"]

    @property
    def weight_config(self) -> Union[Dict[str, Any], None]:
        return self._quant_config["model_config"]["weight"]

    @property
    def experimental_config(self) -> Union[Dict[str, Any], None]:
        return self._quant_config["model_config"]["experimental"]

    @property
    def modelwise_search(self) -> Union[Dict[str, Any], None]:
        return self._quant_config["model_config"]["modelwise_search"]

    @property
    def layerwise_search(self) -> Union[Dict[str, Any], None]:
        return self._quant_config["model_config"]["layerwise_search"]

    @property
    def model_config(self) -> Union[Dict[str, Any], None]:
        return self._quant_config["model_config"]

    @property
    def subgraph_config(self) -> Union[Dict[str, Any], None]:
        return self._quant_config["subgraph_config"]

    @property
    def op_config(self) -> Union[Dict[str, Any], None]:
        return self._quant_config["op_config"]

    @property
    def node_config(self) -> Union[Dict[str, Any], None]:
        return self._quant_config["node_config"]

    def _remove_comments(self, json_like):
        # 移除单行注释
        json_like = re.sub(r"//.*", "", json_like)
        # 移除多行注释
        json_like = re.sub(r"/\*.*?\*/", "", json_like, flags=re.DOTALL)
        return json_like  # noqa: RET504

    def _get_subgraph_nodes(
        self, optimized_model: OnnxModel, subgraph_config: Dict[str, Any]
    ) -> List[OnnxNode]:
        input_nodes = subgraph_config.get("inputs", [])
        output_nodes = subgraph_config.get("outputs", [])
        if input_nodes and output_nodes:
            input_vars = []
            output_vars = []
            for node in optimized_model.graph.nodes:
                if node.name in input_nodes:
                    input_vars.extend(input_var.name for input_var in node.inputs)
                if node.name in output_nodes:
                    output_vars.extend(output_var.name for output_var in node.outputs)
            sub_model = extract_submodel(optimized_model, input_vars, output_vars)
        else:
            raise ValueError("The inputs and outputs of subgraph must be provided.")
        return list(sub_model.graph.nodes)

    def _load_default_quant_config(self, optimized_model: OnnxModel) -> None:
        if not self.activation_config.get(
            "calibration_type"
        ) and not self.model_config.get("layerwise_search"):
            logging.info(
                "Activation calibration type not found, set "
                "multiple calibration methods by default.",
            )
            self.activation_config["calibration_type"] = ["max", "kl"]
            self.activation_config["max_percentile"] = [0.99995, 1.0]
            self.activation_config["per_channel"] = [True, False]
            self.activation_config["asymmetric"] = [True, False]
            self.modelwise_search["similarity"] = 0.995
        if "bias_correction" in self.weight_config:
            bias_correction = self.weight_config["bias_correction"]
            bias_correction["num_sample"] = bias_correction.get("num_sample", 1)
            bias_correction["metric"] = bias_correction.get(
                "metric", "cosine-similarity"
            )
            self.weight_config["bias_correction"] = bias_correction
        if "modelwise_search" in self.model_config and not self.model_config.get(
            "layerwise_search"
        ):
            modelwise_search = self.model_config["modelwise_search"]
            modelwise_search["metric"] = modelwise_search.get(
                "metric", "cosine-similarity"
            )
            self.model_config["modelwise_search"] = modelwise_search
        if "layerwise_search" in self.model_config:
            layerwise_search = self.layerwise_search
            layerwise_search["metric"] = layerwise_search.get(
                "metric", "cosine-similarity"
            )
            self.model_config["layerwise_search"] = layerwise_search
        # set default datatype on target nodes
        if optimized_model is not None and self._march not in [
            "bernoulli",
            "bernoulli2",
        ]:
            # TopK支持输入int8/int16,默认int16
            # HzFilter支持输入int8/int16,默认int16
            for node in optimized_model.graph.nodes:
                if (
                    node.op_type in ["TopK", "HzFilter"]
                    and "all_node_type" not in self.model_config
                    and node.op_type not in self.op_config
                    and node.name not in self.node_config
                ):
                    self.node_config[node.name]["InputType"] = "int16"

    def _load_op_config(self, op_config: Dict[str, Any]) -> None:
        self.op_config.update(op_config)
        # json文件中的"qtype"代表输入数据类型, 需要解析成"InputType"
        for op_config in self.op_config.values():
            qtype = op_config.pop("qtype", None)
            if qtype is not None:
                op_config["InputType"] = qtype

    def _load_subgraph_config(
        self, subgraph_config: Dict[str, Any], optimized_model: OnnxModel
    ) -> None:
        self.subgraph_config.update(subgraph_config)
        # 解析subgraph_config, 生成节点配置项
        if optimized_model is not None:
            for subgraph_config in self.subgraph_config.values():
                nodes = self._get_subgraph_nodes(optimized_model, subgraph_config)
                qtype = subgraph_config.get("qtype", "int8")
                for node in nodes:
                    self.node_config[node.name]["InputType"] = qtype

    def _load_node_config(self, node_config: Dict[str, Any]) -> None:
        self.node_config.update(node_config)
        # 解析node_config, 更新上述生成的节点配置项
        for node_config in self.node_config.values():
            qtype = node_config.pop("qtype", None)
            if qtype is not None:
                node_config["InputType"] = qtype

    def load_user_quant_config(
        self,
        user_quant_config: Union[str, Dict[str, Any]],
        optimized_model: OnnxModel = None,
    ) -> None:
        if user_quant_config:
            if isinstance(user_quant_config, dict):
                pass
            elif isinstance(user_quant_config, str):
                # 从json文件加载quant_config参数。
                with open(user_quant_config) as f:
                    content = f.read()
                    content_no_comments = self._remove_comments(content)
                    user_quant_config = json.loads(content_no_comments)

            # 解析user_quant_config参数
            if "model_config" in user_quant_config:
                self.model_config.update(user_quant_config["model_config"])

            if "op_config" in user_quant_config:
                self._load_op_config(user_quant_config["op_config"])

            if "subgraph_config" in user_quant_config:
                self._load_subgraph_config(
                    user_quant_config["subgraph_config"], optimized_model
                )

            if "node_config" in user_quant_config:
                self._load_node_config(user_quant_config["node_config"])

        self._load_default_quant_config(optimized_model)
        self.check_quant_config()

    def print_quant_config(self):
        config_str = []
        # print model config
        for name, config in self.model_config.items():
            if name == "all_node_type" and config:
                config_str.append(
                    f"All nodes in the model are set to datatype: {config}"
                )
            if name == "model_output_type" and config:
                config_str.append(
                    f"The output nodes of model are set to datatype: {config}"
                )
            if (
                name == "activation"
                and config
                and not self.model_config.get("layerwise_search")
            ):
                config_str.append("The activation calibration parameters:")
                for key, value in config.items():
                    config_str.append("    {:<21} {}".format(key + ":", str(value)))
            if name == "weight" and config:
                config_str.append("The weight calibration parameters:")
                for key, value in config.items():
                    config_str.append("    {:<21} {}".format(key + ":", str(value)))
            if (
                name == "modelwise_search"
                and config
                and not self.model_config.get("layerwise_search")
            ):
                config_str.append("The modelwise search parameters:")
                for key, value in config.items():
                    config_str.append("    {:<21} {}".format(key + ":", str(value)))
            if name == "layerwise_search" and config:
                config_str.append("The layerwise search parameters:")
                for key, value in config.items():
                    config_str.append("    {:<21} {}".format(key + ":", str(value)))
        # print op config
        for op_name, op_attr in self.op_config.items():
            if "InputType" in op_attr:
                config_str.append(
                    "The input of all {} nodes are set to : {}".format(
                        op_name,
                        self.op_config[op_name]["InputType"],
                    ),
                )
            if "OutputType" in op_attr:
                config_str.append(
                    "The output of all {} nodes are set to : {}".format(
                        op_name,
                        self.op_config[op_name]["OutputType"],
                    ),
                )
        # print node config
        for node_name, node_attr in self.node_config.items():
            if "InputType" in node_attr:
                config_str.append(
                    "The input of node {} are set to : {}".format(
                        node_name,
                        self.node_config[node_name]["InputType"],
                    ),
                )
            if "InputType0" in node_attr:
                config_str.append(
                    "The first input of node {} are set to : {}".format(
                        node_name,
                        self.node_config[node_name]["InputType0"],
                    ),
                )
            if "InputType1" in node_attr:
                config_str.append(
                    "The second input of node {} are set to : {}".format(
                        node_name,
                        self.node_config[node_name]["InputType1"],
                    ),
                )
            if "OutputType" in node_attr:
                config_str.append(
                    "The output of node {} are set to : {}".format(
                        node_name,
                        self.node_config[node_name]["OutputType"],
                    ),
                )
        logging.info("\n".join(config_str))

    def check_quant_config(self):
        # 1. 检查一级目录
        for key in self._quant_config:
            if key not in [
                "model_config",
                "subgraph_config",
                "op_config",
                "node_config",
            ]:
                raise ValueError(
                    f"Unsupported first-level title in quant_config: {key}",
                )
        # 2. 检查二级目录
        for key in self.model_config:
            if key not in [
                "all_node_type",
                "model_output_type",
                "activation",
                "weight",
                "modelwise_search",
                "layerwise_search",
                "experimental",
            ]:
                raise ValueError(f"Unsupported config key in model_config: {key}")
        # 3. 检查三级目录
        for op_config in self.op_config.values():
            for key in op_config:
                if key not in ["InputType", "InputType0", "InputType1", "OutputType"]:
                    raise ValueError(f"Unsupported config key in op_config: {key}")
        for node_config in self.node_config.values():
            for key in node_config:
                if key not in [
                    "InputType",
                    "OutputType",
                    "InputType0",
                    "InputType1",
                ]:
                    raise ValueError(f"Unsupported config key in node_config: {key}")
        # 4. 检查数据类型
        self._check_datatype(self._quant_config)
        # 5. 检查校准配置
        self._check_cali_config()

    def _check_datatype(self, item):
        # 根据march对配置的数据类型进行检查
        for key, value in item.items():
            if key in [
                "all_node_type",
                "model_output_type",
                "InputType",
                "OutputType",
                "InputType0",
                "InputType1",
            ] and isinstance(value, str):
                if key == "model_output_type" and value not in [
                    "int8",
                    "int16",
                    "float32",
                ]:
                    raise ValueError(
                        f"Unsupported datatype settings for "
                        f"model_output_type: {value}",
                    )
                if self._march in ["bernoulli", "bernoulli2"]:
                    if value not in ["int8", "float32"]:
                        raise ValueError(
                            f"Unsupported datatype settings for "
                            f"march bernoulli/bernoulli2: {value}",
                        )
                elif self._march in ["bayes", "bayes-e"]:
                    if value not in ["int8", "int16", "float32"]:
                        raise ValueError(
                            f"Unsupported datatype settings for "
                            f"march bayes: {value}",
                        )
                elif self._march in ["nash"] and value not in [
                    "int8",
                    "int16",
                    "float32",
                    "float16",
                ]:
                    raise ValueError(
                        f"Unsupported datatype settings for march nash: {value}",
                    )
                elif self._march in ["b40"] and value not in [
                    "int8",
                    "int16",
                    "float32",
                    "float16",
                    "bfloat16",
                    "float8e4m3fn",
                    "float8e5m2",
                    "float8e3m4fn",
                    "float8e2m5fn",
                    "mxint8",
                ]:
                    # float8, bfloat16和mx这三种数据类型是实验性质的, 不对外释放
                    raise ValueError(
                        f"Unsupported datatype settings for march b40: {value}",
                    )
            elif isinstance(value, (defaultdict, dict)):
                self._check_datatype(value)

    def _check_num_bin(self):
        """检查num_bin参数是否非法."""
        num_bin = self.activation_config.get("num_bin")
        if num_bin is not None:
            check_type(num_bin, item_type=int, list_type=int)
            if isinstance(num_bin, int):
                _num_bin = max(num_bin, 129)
            else:
                _num_bin = [max(_, 129) for _ in num_bin]
            _num_bin = remove_duplicates(_num_bin)
            if num_bin != _num_bin:
                logging.warning(
                    "The num_bin that you set is too small, must be larger "
                    f"than 128! It is already automatically set to {_num_bin}!"
                )
            self.activation_config["num_bin"] = _num_bin

    def _check_max_num_bin(self):
        """检查max_num_bin参数是否非法."""
        max_num_bin = self.activation_config.get("max_num_bin", 16384)
        num_bin = self.activation_config.get("num_bin", 1024)
        check_type(max_num_bin, item_type=int)
        if isinstance(num_bin, int):
            _max_num_bin = max(max_num_bin, num_bin, 258)
        else:
            _max_num_bin = max(max_num_bin, max(num_bin), 258)
        _max_num_bin = remove_duplicates(_max_num_bin)
        if max_num_bin != _max_num_bin:
            logging.warning(
                "The max_num_bin that you set is too small, must be no less than "
                f"258 or num_bin! It is already automatically set to {_max_num_bin}!"
            )
            self.activation_config["max_num_bin"] = _max_num_bin

    def _check_max_percentile(self):
        """检查max_percentile参数是否非法."""
        max_percentile = self.activation_config.get("max_percentile")
        if max_percentile is not None:
            check_type(max_percentile, item_type=float, list_type=float)
            if isinstance(max_percentile, float):
                _max_percentile = max(max_percentile, 0.5)
            else:
                _max_percentile = [max(_, 0.5) for _ in max_percentile]
            _max_percentile = remove_duplicates(_max_percentile)
            if max_percentile != _max_percentile:
                logging.warning(
                    "The max_percentile that you set is too small, must be no less "
                    f"than 0.5! It is already automatically set to {_max_percentile}!"
                )
            self.activation_config["max_percentile"] = _max_percentile

    def _check_per_channel(self):
        """检查per_channel参数是否非法."""
        per_channel = self.activation_config.get("per_channel")
        if per_channel is not None:
            check_type(per_channel, item_type=bool, list_type=bool)
            self.activation_config["per_channel"] = remove_duplicates(per_channel)

    def _check_asymmetric(self):
        """检查asymmetric参数是否非法."""
        asymmetric = self.activation_config.get("asymmetric")
        if asymmetric is not None:
            check_type(asymmetric, item_type=bool, list_type=bool)
            self.activation_config["asymmetric"] = remove_duplicates(asymmetric)

    def _check_cali_config(self):
        """检查校准参数配置."""
        self._check_num_bin()
        self._check_max_num_bin()
        self._check_max_percentile()
        self._check_per_channel()
        self._check_asymmetric()
