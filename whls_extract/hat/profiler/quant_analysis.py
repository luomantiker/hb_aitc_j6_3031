import copy
import logging
import os

import torch

from hat.core.compose_transform import Compose
from hat.registry import OBJECT_REGISTRY, Registry
from hat.utils.package_helper import check_packages_available, require_packages

try:
    from horizon_plugin_profiler import (  # noqa E402
        QuantAnalysis as HorizonQuantAnalysis,
    )
except ImportError:
    HorizonQuantAnalysis = None

logger = logging.getLogger(__name__)

QUANTANALYSIS_REGISTER = Registry("QUANT_ANALYSIS_REGISTRY")


@OBJECT_REGISTRY.register
class QuantAnalysis:
    """Quantization models precision analysis.

    This class helps analyze quantization precision errors between two models.

    Args:
        model: origin float model
        baseline_model_convert_pipeline: baseline model convert pipeline
        analysis_model_convert_pipeline: analysis model convert pipeline
        analysis_model_type: the low precision model type. Support two types:
        - "fake_quant": the analysis_model can be calibration/qat model
            and the baseline_model can be float model or a int8/16 mixed
            qconfig model with good precision.
        - "quantized": the analysis_model must be quantized model and the
            baseline_model must be calibration/qat model.
        out_dir: path to save advisor and comparsion result.
        post_process: post process function which performs on model output.
        data_generator: input dataloader or a custom iterable object. It
            must generate a data each iteration. Example:
        - torch dataloader: data_generator = torch.utils.data.DataLoader()
        - a custom generator: data_generator = [x for x in [1, 2, 3]]
        batch_transforms: batch transforms
        num_steps: num of steps to find bad case
        device_id: device id
        bad_case_input: manually set bad case. If set, will skip auto find
            bad case process.
        analysis_pipeline: analysis pipeline. It is a list of analysis
            operations to do. If None, will do default analysis pipelines.
    """

    @require_packages("horizon_plugin_profiler>=2.1.6")
    def __init__(
        self,
        model: torch.nn.Module,
        baseline_model_convert_pipeline,
        analysis_model_convert_pipeline,
        analysis_model_type,
        out_dir,
        post_process=None,
        dataloader=None,
        batch_transforms=None,
        num_steps=None,
        device_id=None,
        bad_case_input=None,
        analysis_pipeline=None,
    ):
        self.baseline_model = baseline_model_convert_pipeline(
            copy.deepcopy(model)
        )
        self.analysis_model = analysis_model_convert_pipeline(
            copy.deepcopy(model)
        )
        self.device_id = device_id
        self.dataloader = dataloader
        self.bad_case_input = bad_case_input
        self.analysis_pipeline = analysis_pipeline
        self.num_steps = num_steps
        self.out_dir = out_dir
        self.post_process = post_process

        if batch_transforms:
            if isinstance(batch_transforms, (list, tuple)):
                batch_transforms = Compose(batch_transforms)
        self.batch_transforms = batch_transforms

        default_pipeline = [
            dict(  # noqa C408
                analysis_type="AutoFindBadCase",
                data_generator=[
                    self.batch_transforms(x) for x in self.dataloader
                ]
                if self.batch_transforms is not None
                else self.dataloader,
                num_steps=self.num_steps,
                metric="L1",
                device=self.device_id,
                custom_metric_func=None,
                custom_metric_order_seq=None,
            ),
            dict(  # noqa C408
                analysis_type="Run",
                device=self.device_id,
            ),
            dict(  # noqa C408
                analysis_type="ComparePerLayer",
            ),
            dict(  # noqa C408
                analysis_type="Sensitivity",
                device=self.device_id,
                metric="L1",
                reverse=False,
            ),
        ]
        if self.analysis_pipeline is None:
            self.analysis_pipeline = default_pipeline

        if check_packages_available(
            "horizon_plugin_profiler<=2.2.4", raise_exception=False
        ):
            self.quant_analysis = HorizonQuantAnalysis(
                baseline_model=self.baseline_model,
                analysis_model=self.analysis_model,
                analysis_model_type=analysis_model_type,
                post_process=self.post_process,
                out_dir=self.out_dir,
            )
        else:
            self.quant_analysis = HorizonQuantAnalysis(
                baseline_model=self.baseline_model,
                analysis_model=self.analysis_model,
                analysis_model_type=analysis_model_type,
                device_ids=self.device_id,
                post_process=self.post_process,
                out_dir=self.out_dir,
            )

    @QUANTANALYSIS_REGISTER.register
    @QUANTANALYSIS_REGISTER.alias("AutoFindBadCase")
    def _auto_find_bad_case(self, pipeline):
        if self.bad_case_input is not None:
            logger.info("Directly use bad case input. Skip auto find.")
            self.quant_analysis.set_bad_case(self.bad_case_input)
        else:
            if "device" not in pipeline:
                pipeline["device"] = self.device_id
            logger.info("Auto find bad case...")
            self.quant_analysis.auto_find_bad_case(**pipeline)
            logger.info(f"Save bad case in {self.out_dir}.")
            self.quant_analysis.save_bad_case()

    @QUANTANALYSIS_REGISTER.register
    @QUANTANALYSIS_REGISTER.alias("SetModelProfilerDir")
    def _set_model_profiler_dir(self, pipeline):
        logger.info("Set model profiler dir...")
        self.quant_analysis.set_model_profiler_dir(**pipeline)

    @QUANTANALYSIS_REGISTER.register
    @QUANTANALYSIS_REGISTER.alias("LoadBadCase")
    def _load_bad_case(self, pipeline):
        filename = (
            os.path.join(self.out_dir, "badcase.pt")
            if pipeline.get("filename", None) is None
            else pipeline["filename"]
        )
        logger.info(f"Load bad case from {filename}")
        self.quant_analysis.load_bad_case(filename)

    @QUANTANALYSIS_REGISTER.register
    @QUANTANALYSIS_REGISTER.alias("Run")
    def _run(self, pipeline):
        logger.info("Run model...")
        device = pipeline.get("device", self.device_id)
        self.quant_analysis.run(device)

    @QUANTANALYSIS_REGISTER.register
    @QUANTANALYSIS_REGISTER.alias("ComparePerLayer")
    def _run_and_compare_per_layer(self, pipeline):
        logger.info("Compare per layer...")
        self.quant_analysis.compare_per_layer()

    @QUANTANALYSIS_REGISTER.register
    @QUANTANALYSIS_REGISTER.alias("Sensitivity")
    def _sensitivity(self, pipeline):
        logger.info("Run sensitivity analysis...")
        if "device" not in pipeline:
            pipeline["device"] = self.device_id
        self.quant_analysis.sensitivity(**pipeline)

    @QUANTANALYSIS_REGISTER.register
    @QUANTANALYSIS_REGISTER.alias("Clean")
    def _clean(self, pipeline):
        logger.info("Clean...")
        self.quant_analysis.clean()

    def __call__(self):
        for pipeline in self.analysis_pipeline:
            pipeline_type = pipeline.pop("analysis_type", None)
            try:
                QUANTANALYSIS_REGISTER.get(pipeline_type)(self, pipeline)
            except KeyError:
                raise ValueError(f"Unknown analysis type {pipeline_type}")
            except Exception as e:
                raise e
