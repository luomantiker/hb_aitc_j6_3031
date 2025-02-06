# Copyright (c) Horizon Robotics. All rights reserved.

import logging
import os
from abc import abstractmethod
from typing import List

import horizon_plugin_pytorch as horizon
import torch.nn as nn
from horizon_plugin_pytorch.nn.qat import Softmax as qat_softmax

from hat.registry import OBJECT_REGISTRY
from hat.utils import qconfig_manager
from hat.utils.logger import MSGColor, format_msg
from hat.utils.model_helpers import (
    match_children_modules_by_name,
    match_children_modules_by_regex,
)
from hat.utils.package_helper import (
    check_packages_available,
    raise_error_if_import_failed,
)

try:
    import horizon_plugin_profiler
except ImportError:
    horizon_plugin_profiler = None

__all__ = [
    "Float2QAT",
    "Float2Calibration",
    "QATFusePartBN",
    "QAT2Quantize",
    "RepModel2Deploy",
]

logger = logging.getLogger(__name__)


def _preserve_qat_mode(model, preserve_dict):
    if preserve_dict is not None:
        if check_packages_available(
            "horizon_plugin_profiler", raise_exception=False
        ):
            from horizon_plugin_profiler import set_preserve_qat_mode
        elif check_packages_available(
            "horizon_plugin_pytorch>=1.2.0", raise_exception=False
        ):
            from horizon_plugin_pytorch.utils.quant_profiler import (
                set_preserve_qat_mode,
            )
        else:
            raise_error_if_import_failed(
                horizon_plugin_profiler, "horizon_plugin_profiler"
            )

        prefixes = preserve_dict.get("prefixes", ())
        types = preserve_dict.get("types", ())
        set_preserve_qat_mode(model, prefixes, types)


class BaseConverter(object):
    """Base class for defining the process of model convert.

    Args:
        convert_mode: convert mechanism, must be one of ("eager", "symbolic",
            "jit", "jit-strip").
    """

    def __init__(self, convert_mode: str = "eager"):
        if convert_mode == "fx":
            convert_mode = "symbolic"
            logger.info("convert_mode 'fx' has been renamed to 'symbolic'!")

        assert convert_mode in ("eager", "symbolic", "jit", "jit-strip",), (
            f"convert_mode must be one of 'eager', 'fx', 'jit' and 'jit-strip'"
            f", but receive {convert_mode}!"
        )

        if convert_mode == "fx":
            check_packages_available("horizon_plugin_pytorch>=1.0.3")
        elif convert_mode in ("jit", "jit-strip"):
            check_packages_available("horizon_plugin_pytorch>2.3.5")

        self.convert_mode = convert_mode

    @abstractmethod
    def __call__(self, model):
        raise NotImplementedError


@OBJECT_REGISTRY.register
class Float2QAT(BaseConverter):
    """Define the process of convert float model to qat model.

    Args:
        convert_mode: convert mechanism, can be choosen from ('eager',
            'symbolic', 'jit', 'jit-strip').
        hybrid: only used when convert_mode == 'symbolic', please refer to the
            doc of `horizon.quantization.prepare_qat_fx` for more info.
        hybrid_dict: only used when convert_mode == 'symbolic', please refer to
            the doc of `horizon.quantization.prepare_qat_fx` for more info.
        optimize_graph: whether to do some process on origin model for special
            purpose. Currently only support using torch.fx to fix cat input
            scale(only used on Bernoulli).
        qconfig_setter: set qconfig automatically. Value is an qconfig setter
            in horizon_plugin_pytorch.quantization.qconfig_template.
        example_inputs: example inputs for tracing graph. When using
            'jit'/'jit-strip' convert_mode or template qconfig setter, one of
            example_inputs and example_data_loader should be provided.
        example_data_loader: example data loader to get example inputs for
            tracing graph. When using 'jit'/'jit-strip' convert_mode or
            template qconfig setter, one of example_inputs and
            example_data_loader should be provided.
        batch_transforms: batch transforms on example data loader.
        state: model state when tracing.can be choosen from ('train', 'val').
    """

    def __init__(
        self,
        convert_mode="eager",
        hybrid=False,
        hybrid_dict=None,
        optimize_graph=False,
        qconfig_setter=None,
        example_inputs=None,
        example_data_loader=None,
        batch_transforms=None,
        state="val",
    ):
        super(Float2QAT, self).__init__(convert_mode)
        self.hybrid = hybrid
        self.hybrid_dict = hybrid_dict
        self.optimize_graph = optimize_graph
        self.qconfig_setter = qconfig_setter
        self.example_inputs = example_inputs
        self.example_data_loader = example_data_loader
        self.transforms = batch_transforms
        self.state = state

        assert not (
            example_inputs is not None and example_data_loader is not None
        ), (
            "'example_inputs' and 'example_data_loader' can't be provided at"
            " the same time."
        )
        if self.example_data_loader is not None:
            self.example_inputs = next(iter(self.example_data_loader))
        if self.transforms is not None:
            self.example_inputs = self.transforms(self.example_inputs)

        self.use_unified_prepare = (
            hasattr(horizon.quantization, "prepare")
            and not hybrid
            and hybrid_dict is None
            and not optimize_graph
            and convert_mode in ("jit", "jit-strip")
        )
        self.use_qconfig_template = (
            self.qconfig_setter is not None and self.example_inputs is not None
        )

    def __call__(self, model):
        if self.convert_mode == "eager":
            # make sure the input model is a float model
            model.fuse_model()

        qconfig_manager.set_qconfig_mode(qconfig_manager.QconfigMode.QAT)
        if not self.use_qconfig_template:
            model.qconfig = qconfig_manager.get_default_qconfig()
        if hasattr(model, "set_qconfig"):
            model.set_qconfig()
        elif not self.use_qconfig_template:
            raise RuntimeError(
                "use qconfig template or implement `set_qconfig()` in model"
            )

        if self.use_unified_prepare:
            method = None
            for i in horizon.quantization.PrepareMethod:
                if i.value == self.convert_mode:
                    method = i
            if self.state == "val":
                # set eval before prepare to get eval graph.
                origin_training_state = model.training
                model.eval()
            model = horizon.quantization.prepare(
                model,
                example_inputs=self.example_inputs,
                qconfig_setter=self.qconfig_setter,
                method=method,
            )
            if self.state == "val":
                model.train(origin_training_state)

        elif self.convert_mode == "eager":
            horizon.quantization.prepare_qat(
                model,
                inplace=True,
                optimize_graph=self.optimize_graph,
            )
        else:
            model = horizon.quantization.prepare_qat_fx(
                model,
                hybrid=self.hybrid,
                hybrid_dict=self.hybrid_dict,
            )
        logger.info(
            format_msg(
                "Successfully convert float model to qat model.",
                MSGColor.GREEN,
            )
        )
        horizon.quantization.set_fake_quantize(
            model, horizon.quantization.FakeQuantState.QAT
        )
        return model


@OBJECT_REGISTRY.register
class QATFusePartBN(BaseConverter):
    """Define the process of fusing bn in a QAT model.

    Usually used in step fuse bn. Note that module do fuse bn only when
    block implement block."fuse_method"().

    Args:
        qat_fuse_patterns: Regex, compile by re.
        fuse_method: Fuse bn method that block calls.
        regex: Whether to match by regex. if not, match by module name.
        strict: Whether the regular expression is required to be all matched.
    """

    def __init__(
        self,
        qat_fuse_patterns: List[str],
        fuse_method: str = "fuse_norm",
        regex: bool = True,
        strict: bool = False,
    ):
        self.qat_fuse_patterns = qat_fuse_patterns
        self.fuse_method = fuse_method
        self.regex = regex
        self.strict = strict
        super(QATFusePartBN, self).__init__()

    def _fuse_bn(self, model: nn.Module):
        if hasattr(model, self.fuse_method):
            return getattr(model, self.fuse_method)()
        else:
            names = []
            for n, m in model.named_children():
                names.append(n)
                setattr(model, n, self._fuse_bn(m))
            return model

    @property
    def get_match_method(self):
        if self.regex:
            return match_children_modules_by_regex
        else:
            return match_children_modules_by_name

    def __call__(self, model):
        # check qat mode in with bn.
        assert horizon.qat_mode.get_qat_mode() in [
            "with_bn",
            "with_bn_reverse_fold",
        ], (
            f"QATFusePartBN only support in with bn mode."
            f"But get {horizon.qat_mode.get_qat_mode()}"
        )

        gen = self.get_match_method
        for n, m in gen(model, self.qat_fuse_patterns, strict=self.strict):
            setattr(model, n, self._fuse_bn(m))

        logger.info(
            format_msg(
                "Successfully qat float model to qat fuse bn model.",
                MSGColor.GREEN,
            )
        )
        return model


@OBJECT_REGISTRY.register
class Float2Calibration(BaseConverter):
    """Define the process of convert float model to calibration model.

    Args:
        convert_mode: convert mechanism, can be choosen from ('eager',
            'symbolic', 'jit', 'jit-strip').
        hybrid: only used when convert_mode == 'symbolic', please refer to the
            doc of `horizon.quantization.prepare_qat_fx` for more info.
        hybrid_dict: only used when convert_mode == 'symbolic', please refer to
            the doc of `horizon.quantization.prepare_qat_fx` for more info.
        optimize_graph: whether to do some process on origin model for special
            purpose. Currently only support using torch.fx to fix cat input
            scale(only used on Bernoulli).
        qconfig_setter: set qconfig automatically. Value is an qconfig setter
            in horizon_plugin_pytorch.quantization.qconfig_template.
        example_inputs: example inputs for tracing graph. When using
            'jit'/'jit-strip' convert_mode or template qconfig setter, one of
            example_inputs and example_data_loader should be provided.
        example_data_loader: example data loader to get example inputs for
            tracing graph.  When using 'jit'/'jit-strip' convert_mode or
            template qconfig setter, one of example_inputs and
            example_data_loader should be provided.
        batch_transforms: batch transforms on example data loader.
    """

    def __init__(
        self,
        convert_mode="eager",
        hybrid=False,
        hybrid_dict=None,
        optimize_graph=False,
        qconfig_setter=None,
        example_inputs=None,
        example_data_loader=None,
        batch_transforms=None,
    ):
        super(Float2Calibration, self).__init__(convert_mode)
        self.hybrid = hybrid
        self.hybrid_dict = hybrid_dict
        self.optimize_graph = optimize_graph
        self.qconfig_setter = qconfig_setter
        self.example_inputs = example_inputs
        self.example_data_loader = example_data_loader
        self.transforms = batch_transforms

        assert not (
            example_inputs is not None and example_data_loader is not None
        ), (
            "'example_inputs' and 'example_data_loader' can't be provided at"
            " the same time."
        )
        if self.example_data_loader is not None:
            self.example_inputs = next(iter(self.example_data_loader))
        if self.transforms is not None:
            self.example_inputs = self.transforms(self.example_inputs)

        self.use_unified_prepare = (
            hasattr(horizon.quantization, "prepare")
            and not hybrid
            and hybrid_dict is None
            and not optimize_graph
            and convert_mode in ("jit", "jit-strip")
        )
        self.use_qconfig_template = (
            self.qconfig_setter is not None and self.example_inputs is not None
        )

    def __call__(self, model):
        if self.convert_mode == "eager":
            # make sure the input model is a float model
            model.fuse_model()

        qconfig_manager.set_qconfig_mode(
            qconfig_manager.QconfigMode.CALIBRATION
        )
        if not self.use_qconfig_template:
            model.qconfig = qconfig_manager.get_default_qconfig()
        if hasattr(model, "set_qconfig"):
            model.set_qconfig()
        elif not self.use_qconfig_template:
            raise RuntimeError(
                "use qconfig template or implement `set_qconfig()` in model"
            )

        model.eval()
        if self.use_unified_prepare:
            method = None
            for i in horizon.quantization.PrepareMethod:
                if i.value == self.convert_mode:
                    method = i
            model = horizon.quantization.prepare(
                model,
                example_inputs=self.example_inputs,
                qconfig_setter=self.qconfig_setter,
                method=method,
            )
        elif self.convert_mode == "eager":
            horizon.quantization.prepare_qat(
                model,
                inplace=True,
                optimize_graph=self.optimize_graph,
            )
        else:
            model = horizon.quantization.prepare_qat_fx(
                model,
                hybrid=self.hybrid,
                hybrid_dict=self.hybrid_dict,
            )
        logger.info(
            format_msg(
                "Successfully convert float model to calibration model.",
                MSGColor.GREEN,
            )
        )
        horizon.quantization.set_fake_quantize(
            model, horizon.quantization.FakeQuantState.CALIBRATION
        )
        return model


@OBJECT_REGISTRY.register
class QAT2Quantize(BaseConverter):
    """Define the process of convert qat model to quantize model.

    Args:
        convert_mode: convert mechanism, can be choosen from ("eager", "fx").
            if qat model is a hybrid model prepared in "fx" mode, convert_mode
            must be "fx".
        convert_custom_config_dict: only used when convert_mode == 'fx',
            please refer to the doc of `horizon.quantization.convert_fx`
            for more info
        preserve_qat_mode_dict: only used when debugging quantized model
            precision problems. If provided, specified ops in dict preserves
            qat mode in quantized model. This dict format is
                {
                    "prefixes": (tuple, Optional) Set preserve_qat_mode by the
                        prefix of qualified name. Defaults to tuple().
                    "types": (tuple, Optional) Set preserve_qat_mode by module
                        type. Defaults to tuple().
                }
                Eg. preserve_qat_mode_dict={
                    "prefixes": ("model.backbone.conv1", ...)
                    "types": (horizon.nn.qat.Conv2d, ...)
                }
        fast_mode: Whether to accelerate quantized model forward. If set True,
                   quantized model cannot be compiled.
        use_cutlass: Whether to use cutlass accelerate quantized conv.
    """

    def __init__(
        self,
        convert_mode="eager",
        convert_custom_config_dict=None,
        preserve_qat_mode_dict=None,
        fast_mode: bool = False,
        use_cutlass: bool = False,
    ):
        super(QAT2Quantize, self).__init__(convert_mode)
        self.convert_custom_config_dict = convert_custom_config_dict
        self.preserve_qat_mode_dict = preserve_qat_mode_dict
        self.fast_mode = fast_mode
        self.use_cutlass = use_cutlass

    def __call__(self, model):
        kwargs = {}

        if self.fast_mode:
            check_packages_available(
                "horizon_plugin_pytorch>=1.6.3",
                raise_msg="`fast_mode` require `horizon_plugin_pytorch>=1.6.3`",  # noqa E501
            )
            kwargs["fast_mode"] = self.fast_mode

        if self.use_cutlass:
            check_packages_available(
                "horizon_plugin_pytorch>=1.10.1",
                raise_msg="`use_cutlass` require `horizon_plugin_pytorch>=1.10.1`",  # noqa E501
            )
            os.environ["USE_CUTLASS"] = "1"

        _preserve_qat_mode(model, self.preserve_qat_mode_dict)
        # make sure the input model is a qat model
        if self.convert_mode == "eager":
            horizon.quantization.convert(
                model.eval(),
                inplace=True,
                **kwargs,
            )
        else:
            model = horizon.quantization.convert_fx(
                model.eval(),
                **kwargs,
            )
        logger.info(
            format_msg(
                "Successfully convert qat model to quantize model.",
                MSGColor.GREEN,
            )
        )
        return model


@OBJECT_REGISTRY.register
class RepModel2Deploy(BaseConverter):
    """Convert Reparameterized model to deploy mode."""

    def __init__(self):
        super(RepModel2Deploy, self).__init__()

    def __call__(self, model):
        for m in model.modules():
            if hasattr(m, "switch_to_deploy"):
                m.switch_to_deploy()
        return model


@OBJECT_REGISTRY.register
class FixWeightQScale(BaseConverter):
    """Fix qscale of weight while calibration or qat stage."""

    def __init__(self):
        super(FixWeightQScale, self).__init__()

    def __call__(self, model):
        for m in model.modules():
            if hasattr(m, "fix_weight_qscale"):
                m.fix_weight_qscale()
        return model


@OBJECT_REGISTRY.register
class SetSoftMaxDivideStrategy(BaseConverter):
    def __init__(self):
        super(SetSoftMaxDivideStrategy, self).__init__()

    def __call__(self, model):
        for _, m in model.named_modules():
            if isinstance(m, qat_softmax):
                m.reciprocal_kwargs = {"auto_divide_strategy": "evenly"}
        return model
