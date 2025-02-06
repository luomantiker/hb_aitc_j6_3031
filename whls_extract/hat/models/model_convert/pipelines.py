import logging
from typing import Dict, List, Optional

import horizon_plugin_pytorch as horizon

from hat.registry import OBJECT_REGISTRY
from hat.utils import qconfig_manager
from hat.utils.apply_func import _as_list
from .ckpt_converters import LoadCheckpoint
from .converters import (
    BaseConverter,
    Float2Calibration,
    Float2QAT,
    QATFusePartBN,
)

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class ModelConvertPipeline(object):
    """Pipeline multiple converters together to build model.

    Args:
        pipelines: list of ModelConverter or LoadCheckpoint to compose.
        qat_mode: whether need to fuse bn or not.
        qconfig_params: the params of qat config.

    Example:
        model_convert_pipeline = ModelConvertPipeline(
            converters=[
                Float2QAT(),
                LoadCheckpoint(),
                QAT2Quantize(),
            ]
        )
    """

    def __init__(
        self,
        converters: List[BaseConverter],
        qat_mode: Optional[str] = None,
        qconfig_params: Optional[Dict] = None,
    ):
        self.converters = _as_list(converters)
        if qat_mode is None:
            logger.warning('qat_mode is set to "fuse_bn" by default')
        else:
            horizon.qat_mode.set_qat_mode(qat_mode)

        if qconfig_params is None:
            qconfig_params = {}

        self.qconfig_params = qconfig_params
        self.refresh_global_qconfig()

    def refresh_global_qconfig(self):
        qconfig_manager.set_default_qconfig(**self.qconfig_params)

    def __call__(self, model):
        for converter in self.converters:
            if converter is not None:
                model = converter(model)
        return model


@OBJECT_REGISTRY.register
class QATFuseBNConvertPipeline(ModelConvertPipeline):
    """Convert pipeline for QAT Fuse BN case.

    This convert pipeline is created to simplify configurations
    of QAT Fuse BN training. As the name indicates, this pipeline
    works only with QAT training. In each training stage, BatchNorms
    from some user-specfied parts of the whole model are fused
    into nearest Convs.

    This class works closely with QATFusePartBN and LoadCheckpoint
    converter, please refer to the documents for more detail.

    Args:
        qat_mode: whether need to fuse bn or not.
        pre_stage_fuse_patterns: specify which parts of the module should
            be fused in previous stage.
        cur_stage_fuse_patterns: specify which parts of the module should
            be fused in current stage.
        fuse_part_configs: specify the kwargs of QATFuseBNPart converter,
            please refer to its document for details.
        checkpoint_mode: can be "resume" or "pre_step", or left None, when
            no checkpoint provided. "resume" corresponds to the case where
            the provided checkpoint is saved from a module in current
            training stage, while "pre_step" the previous stage. Further
            details of the checkpoint loading (such as how to deal with missed
            or extra parameters in checkpoint) should be specified in
            "checkpoint_configs" arg.
        checkpoint_configs: specify the checkpoint loading details, such as
            checkpoint_path, allow_miss, ignore_extra... During initialization,
            value of this arg is directly passed to LoadCheckpoint converter,
            please refer to its document for details.
        qconfig_params: the params of qat config.
    """

    def __init__(
        self,
        qat_mode: str,
        pre_stage_fuse_patterns: List[BaseConverter],
        cur_stage_fuse_patterns: List[BaseConverter],
        fuse_part_configs: Optional[Dict] = None,
        checkpoint_mode: Optional[str] = None,
        checkpoint_configs: Optional[Dict] = None,
        qconfig_params: Optional[Dict] = None,
    ):
        if checkpoint_mode is not None:
            assert checkpoint_mode in (
                "resume",
                "pre_stage",
            )
            assert checkpoint_configs is not None

        fuse_part_configs = (
            {} if fuse_part_configs is None else fuse_part_configs
        )
        pipelines = [
            Float2QAT(),
            QATFusePartBN(
                qat_fuse_patterns=pre_stage_fuse_patterns,
                **fuse_part_configs,
            ),
            QATFusePartBN(
                qat_fuse_patterns=cur_stage_fuse_patterns,
                **fuse_part_configs,
            ),
        ]

        if checkpoint_mode is not None:
            checkpoint = LoadCheckpoint(**checkpoint_configs)
            idx = (
                len(pipelines)
                if checkpoint_mode == "resume"
                else len(pipelines) - 1
            )
            pipelines.insert(idx, checkpoint)

        super().__init__(
            converters=pipelines,
            qat_mode=qat_mode,
            qconfig_params=qconfig_params,
        )


@OBJECT_REGISTRY.register
class FloatQatConvertPipeline(ModelConvertPipeline):
    """Convert pipeline for QAT Fuse BN case.

    This convert pipeline is created to simplify configurations
    of float-float_freeze_bn-qat training.

    This class works closely with LoadCheckpoint
    converter, please refer to the documents for more detail.

    Args:
        qat_mode: whether need to fuse bn or not.
        enable_qat: whether to convert model to QAT.
        checkpoint_mode: can be "resume" or "pre_step", or left None, when
            no checkpoint provided. "resume" corresponds to the case where
            the provided checkpoint is saved from a module in current
            training stage, while "pre_step" the previous stage. Further
            details of the checkpoint loading (such as how to deal with missed
            or extra parameters in checkpoint) should be specified in
            "checkpoint_configs" arg.
        checkpoint_configs: specify the checkpoint loading details, such as
            checkpoint_path, allow_miss, ignore_extra... During initialization,
            value of this arg is directly passed to LoadCheckpoint converter,
            please refer to its document for details.
        qconfig_params: the params of qat config.
    """

    def __init__(
        self,
        qat_mode: str,
        enable_qat: Optional[bool] = True,
        enable_calibraion: Optional[bool] = False,
        checkpoint_mode: Optional[str] = None,
        checkpoint_configs: Optional[Dict] = None,
        qconfig_params: Optional[Dict] = None,
    ):
        if checkpoint_mode is not None:
            assert checkpoint_mode in (
                "resume",
                "pre_stage",
            )
            assert checkpoint_configs is not None
        pipelines = []
        assert not (enable_qat and enable_calibraion)
        if enable_qat:
            pipelines.append(Float2QAT())
        if enable_calibraion:
            pipelines.append(Float2Calibration())

        if checkpoint_mode is not None:
            checkpoint = LoadCheckpoint(**checkpoint_configs)
            idx = (
                len(pipelines)
                if checkpoint_mode == "resume"
                else len(pipelines) - 1
            )
            pipelines.insert(idx, checkpoint)

        super().__init__(
            converters=pipelines,
            qat_mode=qat_mode,
            qconfig_params=qconfig_params,
        )
