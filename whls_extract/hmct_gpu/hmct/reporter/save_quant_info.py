from collections import defaultdict
import json
from typing import Any, Dict, Optional


def save_quant_info(
    model_quant_type_info: Optional[Dict[str, Dict[str, Any]]] = None,
    node_similarity_info: Optional[Dict[str, Dict[str, Any]]] = None,
    output_similarity_info: Optional[Dict[str, Dict[str, Any]]] = None,
    output_file: str = "quant_info.json",
):
    output_dict: Dict[str, Dict[str, Any]] = defaultdict(dict)
    if model_quant_type_info is not None:
        model_quant_type_info = remove_extra_quant_type_info(model_quant_type_info)
        for node_name in model_quant_type_info:
            # 从model_quant_type_info和node_similarity_info中选择信息写入output_dict中
            # 注意: 输出json中的关键字和集成有耦合, 修改需和集成对接!
            output_dict[node_name]["type"] = model_quant_type_info[node_name]["type"]
            output_dict[node_name]["inputs"] = model_quant_type_info[node_name][
                "input_name"
            ]
            output_dict[node_name]["outputs"] = model_quant_type_info[node_name][
                "output_name"
            ]
            output_dict[node_name]["thresholds"] = model_quant_type_info[node_name][
                "input_threshold"
            ]
            if node_similarity_info is not None and node_name in node_similarity_info:
                output_dict[node_name]["cosine_similarity"] = node_similarity_info[
                    node_name
                ]["Cosine Similarity"]
            else:
                output_dict[node_name]["cosine_similarity"] = "--"

    # 和集成达成约定,模型输出信息单独存在"model_output_tensor"字段中
    if output_similarity_info is not None:
        for output_name in output_similarity_info:
            output_dict["model_output_tensor"][output_name] = {
                "cosine_similarity": output_similarity_info[output_name][
                    "Cosine Similarity"
                ],
            }

    with open(output_file, "w") as f:
        json.dump(output_dict, f, indent=4)


def remove_extra_quant_type_info(
    quant_type_info: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """删除不希望保存到json的quant_type信息."""

    def remove_conv_weight_quant_type_info(
        quant_type_info: Dict[str, Dict[str, Any]],
    ) -> None:
        """删除权重的校准信息."""
        for node_name in quant_type_info:
            if quant_type_info[node_name]["type"] not in ["Conv", "ConvTranspose"]:
                continue
            del quant_type_info[node_name]["input_name"][1:]
            del quant_type_info[node_name]["input_threshold"][1:]
            del quant_type_info[node_name]["input_threshold"][1:]

    remove_conv_weight_quant_type_info(quant_type_info)
    return quant_type_info
