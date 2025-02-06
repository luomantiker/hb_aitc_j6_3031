from abc import ABC, abstractmethod
import logging
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from hmct.ir import OnnxModel, OnnxNode


class PassBase(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError("name Not Implemented for PassBase")

    def __call__(self, onnx_model: "OnnxModel", *args, **kwargs) -> Any:
        self.initialize_pass(onnx_model, *args, **kwargs)
        pass_result = self.run_impl(onnx_model, *args, **kwargs)
        self.finalize_pass(onnx_model, *args, **kwargs)

        return pass_result

    @abstractmethod
    def run_impl(self, onnx_model: "OnnxModel", *args, **kwargs) -> Any:
        raise NotImplementedError("run_impl Not Implemented for PassBase")

    def initialize_pass(self, onnx_model: "OnnxModel", *args, **kwargs) -> None:  # noqa: ARG002
        return

    def finalize_pass(self, onnx_model: "OnnxModel", *args, **kwargs) -> None:  # noqa: ARG002
        return


class PredicateBasedPass(PassBase):
    def run_impl(self, onnx_model: "OnnxModel") -> int:
        num_changes = 0
        for onnx_node in onnx_model.graph.nodes[:]:
            if self.match_pattern(onnx_node):
                num_changes += self.apply_transform(
                    onnx_node=onnx_node, onnx_model=onnx_model
                )

        return num_changes

    @abstractmethod
    def match_pattern(self, onnx_node: "OnnxNode") -> bool:
        raise NotImplementedError(
            "match_pattern Not Implemented for PredicateBasedPass",
        )

    @abstractmethod
    def apply_transform(self, onnx_node: "OnnxNode", onnx_model: "OnnxModel") -> bool:
        raise NotImplementedError(
            "apply_transform Not Implemented for PredicateBasedPass",
        )


def run_optimize_passes(
    onnx_model: "OnnxModel", passes: Sequence[PredicateBasedPass], iterate: bool
) -> "OnnxModel":
    """在onnx模型上执行给定的PredicateBasedPass序列."""
    # initialization before passes run
    onnx_model.infer_shapes()

    iterate_limits = 1024 if iterate else 1
    while iterate_limits > 0:
        intact = True
        for pass_ in passes:
            num_changes = pass_(onnx_model)
            if num_changes > 0:
                intact = False
        if intact:
            break
        iterate_limits -= 1

    if iterate and iterate_limits <= 0:
        logging.warning("It seems an endless iterate to make the passes converge.")

    # finalization after passes run
    onnx_model.infer_shapes().check_validity()

    return onnx_model
