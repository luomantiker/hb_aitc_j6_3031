from collections import defaultdict
from typing import Any, Dict, Optional

from hmct.common import print_info_dict


def print_model_info(
    node_similarity_info: Dict[str, Dict[str, Any]],
    output_similarity_info: Dict[str, Dict[str, Any]],
    model_quant_type_info: Dict[str, Dict[str, Any]],
    model_hybrid_info: Optional[Dict[str, Dict[str, Any]]] = None,
):
    info_display_in_terminal = defaultdict(dict)
    # 存在model_hybrid_info的时候以model_hybrid_info为基准,打印信息包括
    # Node    ON    Subgraph    Type    Cosine Similarity    Threshold    DataType
    if model_hybrid_info:
        for node_name, hybrid_info in model_hybrid_info.items():
            info_display_in_terminal[node_name]["ON"] = hybrid_info["ON"]
            info_display_in_terminal[node_name]["Subgraph"] = hybrid_info["Subgraph"]
            info_display_in_terminal[node_name]["Type"] = hybrid_info["Type"]

            info_display_in_terminal[node_name]["Cosine Similarity"] = "--"
            info_display_in_terminal[node_name]["Threshold"] = "--"
            if node_name in node_similarity_info:
                info_display_in_terminal[node_name]["Cosine Similarity"] = (
                    node_similarity_info[node_name]["Cosine Similarity"]
                )
            if (
                node_name in model_quant_type_info
                and len(model_quant_type_info[node_name]["input_threshold"]) > 0
            ):
                input_threshold = model_quant_type_info[node_name]["input_threshold"][0]
                if input_threshold:
                    info_display_in_terminal[node_name]["Threshold"] = (
                        f"{input_threshold[0]:.6}"
                    )

            info_display_in_terminal[node_name]["DataType"] = hybrid_info["DataType"]
    else:
        # 否则以node_similarity_info为基准,打印信息包括
        # Node    Type    Cosine Similarity    Threshold    DataType
        for node_name, similarity_info in node_similarity_info.items():
            info_display_in_terminal[node_name]["Type"] = model_quant_type_info[
                node_name
            ]["type"]
            info_display_in_terminal[node_name]["Cosine Similarity"] = similarity_info[
                "Cosine Similarity"
            ]
            info_display_in_terminal[node_name]["Threshold"] = "--"
            if len(model_quant_type_info[node_name]["input_threshold"]) > 0:
                input_threshold = model_quant_type_info[node_name]["input_threshold"][0]
                if input_threshold:
                    info_display_in_terminal[node_name]["Threshold"] = (
                        f"{input_threshold[0]:.6}"
                    )
            if len(model_quant_type_info[node_name]["input_qtype"]) > 0:
                info_display_in_terminal[node_name]["DataType"] = model_quant_type_info[
                    node_name
                ]["input_qtype"][0]

    # remove node info when threshold and datatype are both empty.
    def has_threshold_or_datatype(info):
        return info.get("Threshold", "--") != "--" or info.get("DataType", "--") != "--"

    info_display_in_terminal = {
        node: info
        for node, info in info_display_in_terminal.items()
        if has_threshold_or_datatype(info)
    }

    print_info_dict(
        info_display_in_terminal, description="The main quantized node information:"
    )
    print_info_dict(
        output_similarity_info,
        description="The quantized model output:",
        key_type="Output",
    )
