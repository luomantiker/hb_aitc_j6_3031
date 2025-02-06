try:
    from hbdk.torch_script.parser import HBIRBuilder as HB  # noqa N814
    from hbdk.torch_script.parser import ScopeManager as SM  # noqa N817
    from hbdk.torch_script.tools import _preprocess_inputs
    from hbdk.torch_script.utils import TensorManager, check_const_prop

    _HBDK_IMPORTED = True
except ImportError:
    _HBDK_IMPORTED = False

from horizon_plugin_profiler.utils.model_helper import apply_to_collection

import torch


class _ScopeManager(SM):
    """ScopeManager with nodes constant info.

    This class is same as hbdk.torch_script.parser.ScopeManager, in addition to
    maintaining a dict of each node value constant info, which is used to
    determine whether to insert `aten::save` node and in `script_profiler`.
    """

    def __init__(self, name):
        super(_ScopeManager, self).__init__(name)
        self._entire_scope = {}

    def exit_scope(self):
        """Record node constant info after finishing submodule process.

        When a subgraph processed, record the constant info or hbir names of
        the nodes in a large dict.
        """
        for k, v in self._cur_scope.items():
            assert k not in self._entire_scope
            self._entire_scope[k] = (
                None
                if check_const_prop(v)
                else apply_to_collection(
                    v,
                    (TensorManager, torch.Tensor),
                    function=lambda x: x.records[0].hbir
                    if isinstance(x, TensorManager)
                    else None,
                )
            )
        self._scopes.pop()
        self._cur_scope = self._scopes[-1]
        self._scope_name.pop()

    def get_entire_scope(self):
        """Before return, record nodes info of the outermost layer of model."""
        for k, v in self._cur_scope.items():
            assert k not in self._entire_scope
            self._entire_scope[k] = (
                None
                if check_const_prop(v)
                else apply_to_collection(
                    v,
                    dtype=(TensorManager, torch.Tensor),
                    function=lambda x: x.records[0].hbir
                    if isinstance(x, TensorManager)
                    else None,
                )
            )
        return self._entire_scope


class _HBIRBuilder(HB):
    """HBIRBuilder with custom ScopeManager.

    This class is same as hbdk.torch_script.parser.HBIRBuilder, in addition to
    using a custom ScopeManager to record node info into a large dict.
    """

    def __init__(self, func_name, march, run_pred=False):
        super(_HBIRBuilder, self).__init__(func_name, march, run_pred)
        self.scope_man = _ScopeManager([func_name])


def run_hbdk_parser(script_model, example_inputs, march):
    """Run hbdk parser.

    Same as hbdk parser, in addition to using a custom HBIRBuilder to record
    each node info.

    Return:
        Two dicts. One records hbdk parser results, another records node
        constant info.
    """
    if not _HBDK_IMPORTED:
        raise ImportError(
            "script_profiler requires hbdk to compare results, "
            "please install hbdk."
        )
    hbdk_inputs = _preprocess_inputs(example_inputs)
    builder = _HBIRBuilder("", march, run_pred=True)
    builder.build_from_jit(script_model, hbdk_inputs)
    hbdk_results = builder.pred_record
    const_check_map = builder.scope_man.get_entire_scope()
    return hbdk_results, const_check_map
