import pickle

from hmct.common import node_inputs_calibration_relations
from hmct.ir import OnnxModel


class NodeInfo:
    def __init__(self, calibrated_model=None):
        self.node_infos = {}
        self.calibrated_model = calibrated_model
        self.node_to_calibration = {}
        self.calibration_to_node = {}
        self.init()

    def init(self):
        if self.calibrated_model is not None:
            node_to_input_calibration, calibration_to_output_node = (
                node_inputs_calibration_relations(OnnxModel(self.calibrated_model))
            )
            self.node_to_calibration = {
                k: set(v.values()) for k, v in node_to_input_calibration.items()
            }
            self.calibration_to_node = calibration_to_output_node

    def update(self, key, value):
        self.node_infos[key] = value

    def save(self, node_info_file):
        with open(node_info_file, "wb") as f:
            pickle.dump(self.node_infos, f)

    def load(self, node_info_file):
        with open(node_info_file, "rb") as f:
            self.node_infos = pickle.load(f)

    def get(self, entry):
        if entry in self.node_infos:
            return self.node_infos[entry]

        if entry == "node_sensitivity":
            node_info = {}
            if "activation_sensitivity" in self.node_infos:
                for cal_node, value in self.node_infos[
                    "activation_sensitivity"
                ].items():
                    for node in self.calibration_to_node[cal_node]:
                        node_info[node] = value
                self.update("node_sensitivity", node_info)
                return node_info
            if "weight_sensitivity" in self.node_infos:
                for cal_node, value in self.node_infos["weight_sensitivity"].items():
                    for node in self.calibration_to_node[cal_node]:
                        node_info[node] = value
                self.update("node_sensitivity", node_info)
                return node_info
            return None
        return None

    def get_calibration_by_node(self, node_name):
        if node_name in self.node_to_calibration:
            return self.node_to_calibration[node_name]
        return {node_name}
