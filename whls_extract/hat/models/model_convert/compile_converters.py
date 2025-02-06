# Copyright (c) Horizon Robotics. All rights reserved.

import copy
import functools
import logging
from itertools import chain
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn

from hat.registry import OBJECT_REGISTRY
from hat.utils.dynamo import CompileBackendWrapper, disable_compile
from hat.utils.logger import MSGColor, format_msg
from hat.utils.model_helpers import (
    match_children_modules_by_name,
    match_children_modules_by_regex,
)
from hat.utils.package_helper import require_packages
from .converters import BaseConverter

try:
    import torchdynamo
except ImportError:
    torchdynamo = None

__all__ = ["TorchCompile", "Torch2Compile"]


logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class TorchCompile(BaseConverter):
    """Convert torch module to compile wrap module.

    NOTE: Compilation occurs at the first model forward!
    Slower is as expected!

    Args:
        compile_backend: TorchDynamo compile optimizer backend.
        load_extensions: Load extension from hat.utils.trt_fx_extension.py.
    """

    @require_packages("torchdynamo")
    def __init__(
        self,
        compile_backend: Optional[Union[Callable, str]] = None,
        load_extensions: Optional[Union[List[str], str]] = None,
    ):
        super(TorchCompile, self).__init__()
        self.compile_backend = compile_backend
        from hat.utils.trt_fx_extension import load_extension

        if load_extensions is not None:
            if isinstance(load_extensions, str):
                load_extensions = [load_extensions]
            for ext in load_extensions:
                load_extension(ext)

    def __call__(self, model):
        logger.info(
            format_msg(
                f"Wrap torchDynamo optimizer by backend "
                f"{self.compile_backend}...\n"
                f"NOTE: the first time forward is slower due to compilation!",
                MSGColor.RED,
            ),
        )
        model = torchdynamo.optimize(self.compile_backend)(model)

        return model


@OBJECT_REGISTRY.register
class Torch2Compile(BaseConverter):
    """Compile model(nn.Module) by `torch.compile()` in torch>=2.0.

    .. Note::
       compile_submodules and skip_modules are mutually exclusive and
       can only be selected for use. If none of them are used,
       the entire model will be compiled.

    Args:
       compile_submodules: Module to compile, support regex or module name.
       skip_modules: Module to skip compile, support regex or module name.
       regex: Whether to match by regex. if not, match by module name.
       strict: Whether regular expression is required to be all matched.
       dynamo_cfg: A dictionary of options to set `torch._dynamo.config`.
       finegrained_cfg: A dictionary of fine-grained configuration
        for submodules. For example:

        finegrained_cfg = {
            "backbone": {"fullgraph": "True"},
            "neck": {"fullgraph": "False"}}

       kwargs: Args of `torch.compile` interface, see:
       https://pytorch.org/docs/stable/generated/torch.compile.html#torch.compile
    """

    @require_packages("torch>=2.0")
    def __init__(
        self,
        compile_submodules: List[str] = None,
        skip_modules: List[str] = None,
        regex: bool = True,
        strict: bool = False,
        dynamo_cfg: Optional[Dict] = None,
        finegrained_cfg: Optional[Dict[str, Dict[str, str]]] = None,
        **kwargs,
    ):
        super(Torch2Compile, self).__init__()
        self.compile_submodules = compile_submodules
        self.skip_modules = skip_modules
        self.regex = regex
        self.strict = strict
        self.finegrained_cfg = finegrained_cfg
        self.compile_args = kwargs

        default_dynamo_cfg = {}

        if dynamo_cfg is None:
            self.dynamo_cfg = default_dynamo_cfg
        else:
            self.dynamo_cfg = dict(
                chain(default_dynamo_cfg.items(), dynamo_cfg.items())
            )

    def __call__(self, model):
        # set dynamo config
        try:
            from torch import _dynamo as torch_dynamo

            torch_dynamo.reset()

            for k, v in self.dynamo_cfg.items():
                if hasattr(torch_dynamo.config, k):
                    setattr(torch_dynamo.config, k, v)
                else:
                    logger.warning(
                        format_msg(
                            msg=f"`torch._dynamo.config` does not have attr `{k}`, skip set `torch._dynamo.config.{k}={v}`",  # noqa E501
                            color=MSGColor.RED,
                        )
                    )
        except Exception as e:
            logger.warning(
                format_msg(
                    msg=f"Failed to set `torch._dynamo.config`, caused by `{e}`",  # noqa E501
                    color=MSGColor.RED,
                )
            )

        # skip module
        if self.skip_modules:
            model = self.skip_compile_modules(
                model=model,
                skip_modules=self.skip_modules,
                regex=self.regex,
                strict=self.strict,
            )

        logger.info(
            format_msg(
                msg="Using `torch.compile`, which may take a few minutes in first iteration step...",  # noqa E501
                color=MSGColor.GREEN,
            )
        )
        backend = self.compile_args.pop("backend", "inductor")
        backend = CompileBackendWrapper(backend=backend, **self.compile_args)
        if self.compile_submodules:
            model = self.compile_modules(
                model=model,
                compile_submodules=self.compile_submodules,
                regex=self.regex,
                strict=self.strict,
                backend=backend,
                finegrained_cfg=self.finegrained_cfg,
                **self.compile_args,
            )
        else:
            model = torch.compile(model, backend=backend, **self.compile_args)

        logger.info(
            format_msg(
                msg="Wrapped model with `torch.compile()`.",  # noqa E501
                color=MSGColor.GREEN,
            )
        )
        return model

    @staticmethod
    def compile_modules(
        model: nn.Module,
        compile_submodules: List[str],
        regex: bool = True,
        strict: bool = False,
        finegrained_cfg: Optional[Dict[str, Dict[str, str]]] = None,
        **kwargs,
    ):
        """Add a wrap hook to compile submodule.

        Args:
            model: Model to add hook.
            skip_modules: Submodule to compile, support regex or module name.
            regex: Whether to match by regex. if not, match by module name.
            strict: Whether regular expression is required to be all matched.
        """

        def _wrap_forward(forward_func, detial_args):
            _kwargs = copy.deepcopy(kwargs)
            _kwargs.update(detial_args)

            @functools.wraps(forward_func)
            @torch.compile(**_kwargs)
            def wrapper(*args, **a_kwargs):
                output = forward_func(*args, **a_kwargs)
                return output

            return wrapper

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            modules = model.module
        else:
            modules = model

        if regex:
            gen = match_children_modules_by_regex
        else:
            gen = match_children_modules_by_name

        white_list = []
        for n, m in gen(modules, compile_submodules, strict=strict):
            if finegrained_cfg and n in finegrained_cfg:
                m.forward = _wrap_forward(m.forward, finegrained_cfg[n])
            m.forward = _wrap_forward(m.forward, {})
            white_list.append(n)

        logger.info(
            format_msg(
                f"Compile submodules:\n {white_list}.\n",
                MSGColor.GREEN,
            )
        )

        return model

    @staticmethod
    def skip_compile_modules(
        model: nn.Module,
        skip_modules: List[str],
        regex: bool = True,
        strict: bool = False,
    ):
        """Add a wrap hook to skip compile.

        Args:
            model: Model to add hook.
            skip_modules: Module to skip compile, support regex or module name.
            regex: Whether to match by regex. if not, match by module name.
            strict: Whether regular expression is required to be all matched.
        """

        def _wrap_forward(forward_func):
            @functools.wraps(forward_func)
            @disable_compile
            def wrapper(*args, **kwargs):
                output = forward_func(*args, **kwargs)
                return output

            return wrapper

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            modules = model.module
        else:
            modules = model

        if regex:
            gen = match_children_modules_by_regex
        else:
            gen = match_children_modules_by_name

        disabled_list = []
        for n, m in gen(modules, skip_modules, strict=strict):
            m.forward = _wrap_forward(m.forward)
            disabled_list.append(n)
        logger.info(
            format_msg(
                f"Skip compile modules:\n {disabled_list}.\n",
                MSGColor.GREEN,
            )
        )

        return model
