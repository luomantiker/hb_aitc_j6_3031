import contextlib
import io
import logging
import os
from distutils.version import LooseVersion
from functools import wraps

import torch.onnx
from torch import _C  # noqa: N814
from torch.onnx import OperatorExportTypes

from horizon_plugin_pytorch.march import March, get_march
from horizon_plugin_pytorch.utils import deprecated_interface_warning
from horizon_plugin_pytorch.utils._register_onnx_ops import (
    register_all_custom_op_symbolic,
)

TrainingMode = _C._onnx.TrainingMode
logger = logging.getLogger(__name__)

__all__ = ["export_to_onnx", "export_quantized_onnx"]


def _preprocess_graph(func):
    """Remove dead custom registered ops.

    Custom registered quantized functions with a tuple returned usually
    followed by a prim::TupleUnpack node in traced graph. However, if this func
    results is not used by other nodes, there is no prim::TupleUnpack node
    followed to unpack the results. Dead code elimination pass in torch can not
    remove this node(May this custom op is traced as a block and marked as
    live). So find unused custom ops and delete from graph here.
    """

    @wraps(func)
    def _preprocess(*args, **kwargs):
        graph, *args = args
        assert type(graph) == torch._C.Graph
        dead_vslz_node = []
        for node in graph.nodes():
            if node.kind() == "prim::PythonOp" and not node.hasUses():
                dead_vslz_node.append(node)
        for node in dead_vslz_node:
            node.destroy()

        return func(graph, *args, **kwargs)

    return _preprocess


# torch 1.10.2 add some logic in onnx shape inference and use std::cerr
# print warnings in custom registered ops.
# We redirect stderr to null to avoid warnings in each custom op,
# do torch.onnx.export and then redirect stderr back.
@contextlib.contextmanager
def _redirect_stderr():
    # Note: Directly use sys.stderr.fileno() cause 'Tee' error in CI/CD
    # stderr_fd = sys.stderr.fileno()
    stderr_fd = 2
    fd = os.open("/dev/null", os.O_WRONLY)
    dup_stderr_fd = os.dup(stderr_fd)
    try:
        yield os.dup2(fd, stderr_fd)
    finally:
        os.dup2(dup_stderr_fd, stderr_fd)
        os.close(fd)
        os.close(dup_stderr_fd)


# replace torch.onnx.utils._optimize_graph in torch 1.13 to avoid
# process of autograd function inner implementation
@contextlib.contextmanager
def _redirect_opt_graph():
    _torch_optimize_graph = torch.onnx.utils._optimize_graph
    try:
        if LooseVersion(torch.__version__) >= LooseVersion("1.13"):
            from ._optimize_graph_helper import _optimize_graph

            torch.onnx.utils._optimize_graph = _preprocess_graph(
                _optimize_graph
            )
            yield True
        else:
            torch.onnx.utils._optimize_graph = _preprocess_graph(
                _torch_optimize_graph
            )
            yield False
    finally:
        torch.onnx.utils._optimize_graph = _torch_optimize_graph


@contextlib.contextmanager
def _set_is_in_onnx_export_false():
    origin_f = torch.onnx.utils.is_in_onnx_export
    try:
        if LooseVersion(torch.__version__) >= LooseVersion("1.13"):
            torch.onnx.utils.is_in_onnx_export = False
        yield
    finally:
        torch.onnx.utils.is_in_onnx_export = origin_f


def _attach_metadata_to_onnx(input_io, f):
    input_io.flush()
    input_io.seek(0)

    try:
        import onnx
    except ImportError:
        logger.warning("Do not find onnx, we cannot add metadata to onnx file")
        if isinstance(f, str):
            with open(f, "wb") as file:
                file.write(input_io.getbuffer())
        else:
            f.write(input_io.getbuffer())
    else:
        import horizon_plugin_pytorch

        model = onnx.load(input_io)
        meta = model.metadata_props.add()
        meta.key = "produced_by"
        meta.value = "horizon_plugin_pytorch=={}".format(
            horizon_plugin_pytorch.__version__
        )
        onnx.save(model, f)


def export_to_onnx(
    model,
    args,
    f,
    export_params=True,
    verbose=False,
    training=TrainingMode.EVAL,
    input_names=None,
    output_names=None,
    operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
    opset_version=11,
    do_constant_folding=True,
    dynamic_axes=None,
    keep_initializers_as_inputs=None,
    custom_opsets=None,
):
    r"""
    Export a (float or qat)model into ONNX format.

    Args:
        model (torch.nn.Module/torch.jit.ScriptModule/ScriptFunction):
            the model to be exported.
        args (tuple or torch.Tensor):

            args can be structured either as:

            1. ONLY A TUPLE OF ARGUMENTS::

                args = (x, y, z)

               The tuple should contain model inputs such that `model(*args)`
               is a valid invocation of the model. Any non-Tensor arguments
               will be hard-coded into the exported model; any Tensor arguments
               will become inputs of the exported model, in the order they
               occur in the tuple.

            2. A TENSOR::

                args = torch.Tensor([1])

               This is equivalent to a 1-ary tuple of that Tensor.

            3. A TUPLE OF ARGUMENTS ENDING WITH A DICTIONARY OF NAMED
               ARGUMENTS::

                args = (x,
                        {'y': input_y,
                         'z': input_z})

               All but the last element of the tuple will be passed as
               non-keyword arguments, and named arguments will be set from the
               last element.
               If a named argument is not present in the dictionary , it is
               assigned the default value, or None if a default value is not
               provided.

        f: a file-like object or a string containing a file name.  A binary
            protocol buffer will be written to this file.
        export_params (bool, default True): if True, all parameters will
            be exported.
        verbose (bool, default False): if True, prints a description of the
            model being exported to stdout, doc_string will be added to graph.
            doc_string may contaion mapping of module scope to node name in
            future torch onnx.
        training (enum, default TrainingMode.EVAL):
            if model.training is False and in training mode if model.training
            is True.

            * ``TrainingMode.EVAL``: export the model in inference mode.
            * ``TrainingMode.PRESERVE``: export the model in inference mode
            * ``TrainingMode.TRAINING``: export the model in training mode.
              Disables optimizations which might interfere with training.
        input_names (list of str, default empty list): names to assign to the
            input nodes of the graph, in order.
        output_names (list of str, default empty list): names to assign to the
            output nodes of the graph, in order.
        operator_export_type (enum, default ONNX_FALLTHROUGH):

            * ``OperatorExportTypes.ONNX``: Export all ops as regular ONNX ops
              (in the default opset domain).
            * ``OperatorExportTypes.ONNX_FALLTHROUGH``: Try to convert all ops
              to standard ONNX ops in the default opset domain.
            * ``OperatorExportTypes.ONNX_ATEN``: All ATen ops (in the
              TorchScript namespace "aten") are exported as ATen ops.
            * ``OperatorExportTypes.ONNX_ATEN_FALLBACK``: Try to export each
              ATen op (in the TorchScript namespace "aten") as a regular ONNX
              op. If we are unable to do so,fall back to exporting an ATen op.
        opset_version (int, default 11): by default we export the model to the
            opset version of the onnx submodule.
        do_constant_folding (bool, default False): Apply the constant-folding
            optimization. Constant-folding will replace some of the ops that
            have all constant inputs with pre-computed constant nodes.
        dynamic_axes (dict<str, list(int)/dict<int, str>>, default empty dict):
            By default the exported model will have the shapes of all input
            and output tensors set to exactly match those given in ``args``
            (and ``example_outputs`` when that arg is required). To specify
            axes of tensors as dynamic (i.e. known only at run-time), set
            ``dynamic_axes`` to a dict with schema:

            * KEY (str): an input or output name. Each name must also be
              provided in ``input_names`` or ``output_names``.

            * VALUE (dict or list): If a dict, keys are axis indices and
              values are axis names. If a list, each element is an axis index.

        keep_initializers_as_inputs (bool, default None): If True, all the
            initializers (typically corresponding to parameters) in the
            exported graph will also be added as inputs to the graph. If False,
            then initializers are not added as inputs to the graph, and only
            the non-parameter inputs are added as inputs.
            This may allow for better optimizations (e.g. constant folding) by
            backends/runtimes.

        custom_opsets (dict<str, int>, default empty dict):
            A dict with schema:

            * KEY (str): opset domain name
            * VALUE (int): opset version

            If a custom opset is referenced by ``model`` but not mentioned in
            this dictionary, the opset version is set to 1.
    """
    assert not (
        isinstance(model, torch.jit.ScriptModule)
        or isinstance(model, torch.jit.ScriptFunction)
    ), (
        "{} is a ScriptModule or ScriptFunction!!".format(model._get_name())
        + " Only support export quantized torch.nn.Module"
    )

    if get_march() == March.BAYES and opset_version != 11:
        logger.warning(
            "Hybrid qat in bayes march only supports opset 11. If your purpose"
            " is just to visualize the model, please ignore this warning. ",
            extra={"call_times_context": ("message")},
        )

    register_all_custom_op_symbolic(opset_version)

    if not (operator_export_type == OperatorExportTypes.ONNX_FALLTHROUGH):
        logger.warning(
            f"Because some Operations are not supported by ONNX, it may "
            f"fail when using `operator_export_type ={operator_export_type}`."
            f"If an error occurs, please try to use "
            f"`OperatorExportTypes.ONNX_FALLTHROUGH`",
            extra={"call_times_context": ("message")},
        )

    kwargs = {"strip_doc_string": True, "enable_onnx_checker": True}

    tmp_onnx_io = io.BytesIO()

    with _redirect_stderr(), _redirect_opt_graph() as torch_113:
        kwargs = {} if torch_113 else kwargs
        torch.onnx.export(
            model,
            args,
            tmp_onnx_io,
            export_params=export_params,
            verbose=verbose,
            training=training,
            input_names=input_names,
            output_names=output_names,
            operator_export_type=operator_export_type,
            do_constant_folding=do_constant_folding,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            keep_initializers_as_inputs=keep_initializers_as_inputs,
            custom_opsets=custom_opsets,
            **kwargs,
        )

    _attach_metadata_to_onnx(tmp_onnx_io, f)


@deprecated_interface_warning("2.4.7", "2.7.0", export_to_onnx)
def export_quantized_onnx(
    model,
    args,
    f,
    export_params=True,
    verbose=False,
    training=TrainingMode.EVAL,
    input_names=None,
    output_names=None,
    operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
    opset_version=None,
    do_constant_folding=True,
    dynamic_axes=None,
    keep_initializers_as_inputs=None,
    custom_opsets=None,
):
    r"""Export a quantized model into ONNX format.

    Args are same with torch.onnx.export
    """
    register_all_custom_op_symbolic()
    assert not (
        isinstance(model, torch.jit.ScriptModule)
        or isinstance(model, torch.jit.ScriptFunction)
    ), (
        "{} is a ScriptModule or ScriptFunction!!".format(model._get_name())
        + " Only support export quantized torch.nn.Module"
    )

    if not (operator_export_type == OperatorExportTypes.ONNX_FALLTHROUGH):
        logger.warning(
            f"Because some torch Operations are not supported by ONNX, it may "
            f"fail when using `operator_export_type ={operator_export_type}`."
            f"If an error occurs, please try to use "
            f"`OperatorExportTypes.ONNX_FALLTHROUGH`",
            extra={"call_times_context": ("message")},
        )

    kwargs = {"strip_doc_string": True, "enable_onnx_checker": True}

    tmp_onnx_io = io.BytesIO()

    with _redirect_stderr(), _redirect_opt_graph() as torch_113:
        kwargs = {} if torch_113 else kwargs
        torch.onnx.export(
            model,
            args,
            tmp_onnx_io,
            export_params=export_params,
            verbose=verbose,
            training=training,
            input_names=input_names,
            output_names=output_names,
            operator_export_type=operator_export_type,
            opset_version=opset_version,
            do_constant_folding=do_constant_folding,
            dynamic_axes=dynamic_axes,
            keep_initializers_as_inputs=keep_initializers_as_inputs,
            custom_opsets=custom_opsets,
            **kwargs,
        )

    _attach_metadata_to_onnx(tmp_onnx_io, f)
