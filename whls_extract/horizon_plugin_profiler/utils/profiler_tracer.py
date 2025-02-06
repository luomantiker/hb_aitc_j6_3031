import logging

from horizon_plugin_profiler.utils.location_info import LocationManager
from horizon_plugin_profiler.utils.op_running_info import (
    OpRunningInfo,
    OpRunningInfoManager,
)
from horizon_plugin_profiler.utils.typeguard import typechecked

import torch
from torch import Tensor

from .bc_helper import get_single_bc_error
from .model_helper import HookAndTorchFunctionHelper

logger = logging.getLogger(__name__)


class OpInfoRecorder(HookAndTorchFunctionHelper):
    """Getting graph with module hook and __torch_function__ of tensor."""

    # Only for accessing OpInfoRecorder obj in TracerTensor
    _current = None

    class TracerTensor(HookAndTorchFunctionHelper.TracerTensor):
        """Patch Tensor for tracing."""

        @classmethod
        def _torch_function_postprocess(
            cls, func, types, args, kwargs, func_ret
        ):
            """Postprocess of __torch_function__."""
            if OpInfoRecorder.is_tracing():
                if isinstance(func_ret, Tensor) or (
                    isinstance(func_ret, tuple)
                    and isinstance(func_ret[0], Tensor)
                ):
                    OpInfoRecorder._current.info_manager.add(
                        OpRunningInfo(
                            LocationManager.get(func, update_user_stack=True),
                            input=cls.unwrap_to_origin(args),
                            output=cls.unwrap_to_origin(func_ret),
                        )
                    )

            return super()._torch_function_postprocess(
                func, types, args, kwargs, func_ret
            )

    @typechecked
    def __init__(self, model: torch.nn.Module, out_dir: str) -> None:
        self.model = model
        self.out_dir = out_dir
        self.info_manager: OpRunningInfoManager = None
        super(OpInfoRecorder, self).__init__()
        self.hb_profiler = None

    @classmethod
    def current(cls, *args, **kwargs):
        return cls._current

    @classmethod
    def is_tracing(cls):
        return cls._current is not None

    def _forward_hook(self, mod, args, kwargs, output):
        """Implement module forward hook."""
        mod_states = {}
        for name, param in mod.named_parameters():
            mod_states[name] = param
        # skip observer buffers and scale, zero_point
        for name, buf in mod.named_buffers():
            if all([x not in name for x in (".", "scale", "zero_point")]):
                mod_states[name] = buf
        self.info_manager.add(
            OpRunningInfo(
                LocationManager.get(mod),
                input=self.TracerTensor.unwrap_to_origin(args),
                output=self.TracerTensor.unwrap_to_origin(output),
                op_states=mod_states,
            )
        )
        return super()._forward_hook(mod, args, kwargs, output)

    def __enter__(self, *args, **kwargs):
        self.old_obj = OpInfoRecorder._current
        OpInfoRecorder._current = self

        self.info_manager = OpRunningInfoManager(self.out_dir)
        from horizon_plugin_profiler import HbirModelProfiler

        from .model_helper import HbirModuleWrapper

        # Temporary solution: support hbir module in torch.nn.Module.
        # Hbir should be wrapped with HbirModuleWrapper manually.
        for name, mod in self.model.named_modules():
            if isinstance(mod, HbirModuleWrapper):
                self.hb_profiler = HbirModelProfiler(mod, self.out_dir, name)
        if self.hb_profiler is not None:
            self.hb_profiler.__enter__()
            self.hb_profiler.info_manager = self.info_manager

        super().__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        self.info_manager.dump_meta()
        super().__exit__()
        if self.hb_profiler is not None:
            self.hb_profiler.__exit__()
        OpInfoRecorder._current = self.old_obj

    def get_info_manager(self) -> OpRunningInfoManager:
        return self.info_manager


class OpInfoRecorderWithSinglebc(OpInfoRecorder):
    class TracerTensor(OpInfoRecorder.TracerTensor):
        """Patch Tensor for tracing."""

        @classmethod
        def _torch_function_postprocess(
            cls, func, types, args, kwargs, func_ret
        ):
            """Postprocess of __torch_function__."""
            if OpInfoRecorder.is_tracing():
                if isinstance(func_ret, Tensor) or (
                    isinstance(func_ret, tuple)
                    and isinstance(func_ret[0], Tensor)
                ):
                    same_in_output = get_single_bc_error(
                        func, args, kwargs, func_ret
                    )
                    OpInfoRecorder._current.info_manager.add(
                        OpRunningInfo(
                            LocationManager.get(func, update_user_stack=True),
                            input=cls.unwrap_to_origin(args),
                            output=cls.unwrap_to_origin(func_ret),
                            same_in_qatbc_out=same_in_output["qatbc"]
                            if same_in_output is not None
                            else None,
                            same_in_intbc_out=same_in_output["intbc"]
                            if same_in_output is not None
                            else None,
                        )
                    )

            return cls.wrap(func_ret)

    def _forward_hook(self, mod, args, kwargs, output):
        """Implement module forward hook."""
        mod_states = {}
        for name, param in mod.named_parameters():
            mod_states[name] = param
        # skip observer buffers and scale, zero_point
        for name, buf in mod.named_buffers():
            if all([x not in name for x in (".", "scale", "zero_point")]):
                mod_states[name] = buf
        same_in_output = get_single_bc_error(mod, args, kwargs, output)
        self.info_manager.add(
            OpRunningInfo(
                LocationManager.get(mod),
                input=self.TracerTensor.unwrap_to_origin(args),
                output=self.TracerTensor.unwrap_to_origin(output),
                op_states=mod_states,
                same_in_qatbc_out=same_in_output["qatbc"]
                if same_in_output is not None
                else None,
                same_in_intbc_out=same_in_output["intbc"]
                if same_in_output is not None
                else None,
            )
        )
        return self.TracerTensor.wrap(output)
