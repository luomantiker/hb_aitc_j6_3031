from abc import abstractmethod
from typing import TYPE_CHECKING, Iterable, Iterator, List, Union

from hmct.common import PassBase

if TYPE_CHECKING:
    from hmct.ir import OnnxModel


class CalibrationPass(PassBase):
    """CalibrationPass defines common interfaces for different calibration passes.

    This class provides a template for implementing the specific
    calibration pass.
    """

    @abstractmethod
    def run_impl(self, calibrated_model: "OnnxModel", **kwargs) -> "OnnxModel":
        raise NotImplementedError("run_impl Not Implemented for CalibrationPass.")


class CalibrationPipeline(Iterable):
    """CalibrationPipeline stores a list of CalibrationPass.

    It provides methods to add calibration passes to the pipeline,
    calibrate a model using the pipeline, and iterate over the calibration
    passes in the pipeline.
    """

    def __init__(self):
        super().__init__()
        self._pipelines = []

    def set(self, passes: Union[List[CalibrationPass], CalibrationPass]) -> None:
        if isinstance(passes, CalibrationPass):
            passes = [passes]
        for cal_pass in passes:
            assert isinstance(cal_pass, CalibrationPass), (
                "Calibration Pipeline only suppose to contain calibration "
                f"passes, but got {cal_pass!s}({type(cal_pass)})."
            )
        self._pipelines.extend(passes)

    def calibrate(self, calibrated_model: "OnnxModel", **kwargs) -> "OnnxModel":
        for cal_pass in self._pipelines:
            calibrated_model = cal_pass(calibrated_model, **kwargs)
        return calibrated_model

    def __iter__(self) -> Iterator[CalibrationPass]:
        return self._pipelines.__iter__()
