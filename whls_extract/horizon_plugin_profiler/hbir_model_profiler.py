from distutils.version import LooseVersion
from typing import Optional, Union

from horizon_plugin_profiler.utils.hbdk4_optional import HbirModule
from horizon_plugin_profiler.utils.location_info import TorchLocationInfo
from horizon_plugin_profiler.utils.model_helper import HbirModuleWrapper
from horizon_plugin_profiler.utils.op_running_info import (
    OpRunningInfo,
    OpRunningInfoManager,
)
from horizon_plugin_profiler.utils.typeguard import typechecked

import torch


class HbirModelProfiler:
    @typechecked
    def __init__(
        self,
        model: Union[HbirModule, HbirModuleWrapper],
        work_dir: Optional[str] = None,
        prefix: Optional[str] = None,
    ):
        if isinstance(model, HbirModule):
            model = HbirModuleWrapper(model)
        self.func = model.func
        self.prefix = prefix
        self.info_manager = OpRunningInfoManager(work_dir)
        self._enabled = False

        self.func.register_callback(self.hbir_record_callabck)

    def hbir_record_callabck(self, op, results, operands):
        if (
            self._enabled
            and hasattr(op, "track_attr")
            and op.track_attr is not None
        ):
            location_dict = op.track_attr.debug_info
            loc = TorchLocationInfo.from_dict(location_dict)
            if self.prefix is not None:
                loc.mod_name = self.prefix + "." + loc.mod_name

            results = type(results)(torch.from_numpy(x) for x in results)
            operands = type(operands)(torch.from_numpy(x) for x in operands)

            self.info_manager.add(
                OpRunningInfo(
                    loc, input=operands, output=results, hbir_op_type=op.type
                )
            )

        return True

    def __enter__(self, *args, **kwargs):
        self._enabled = True
        return self

    def __exit__(self, *args, **kwargs):
        self.info_manager.dump_meta()

        def empty_callback(op, results, operands):
            return True

        from hbdk4.compiler import version

        if LooseVersion(version.VERSION) >= LooseVersion("4.0.14"):
            self.func.register_callback(empty_callback)

        self._enabled = False

    def get_info_manager(self) -> OpRunningInfoManager:
        return self.info_manager
