import functools
import importlib
import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hmct.ir import OnnxModel

OP_IMPLEMENTS = {}
CURRENT_CONTEXT = None


class Identity:
    def __init__(self, counts):
        self._counts = counts

    def compute(self, *inputs):
        return inputs[-self._counts :]


class CustomOpContext:
    def __init__(self, module):
        self.module = module

    def __enter__(self):
        global CURRENT_CONTEXT
        self.last = CURRENT_CONTEXT
        CURRENT_CONTEXT = self

    def __exit__(self, exc_type, exc_value, traceback):
        global CURRENT_CONTEXT
        CURRENT_CONTEXT = self.last

    def get_module(self):
        return self.module


CURRENT_CONTEXT = CustomOpContext(".")


def get_current_context():
    global CURRENT_CONTEXT
    return CURRENT_CONTEXT


def register_op_as_identity(kind, counts):
    new_identity = functools.partial(Identity, counts=counts)
    setattr(sys.modules[__name__], kind, new_identity)
    if kind not in OP_IMPLEMENTS:
        OP_IMPLEMENTS[kind] = {"cls": new_identity, "module": __name__, "name": kind}


def get_op_implement(kind):
    if kind in OP_IMPLEMENTS:
        return OP_IMPLEMENTS[kind]
    return None


def get_all_op_implements():
    return OP_IMPLEMENTS


def check_module(module_name):
    module_spec = importlib.util.find_spec(module_name)
    if module_spec is None:
        print(f"Module :{module_name} not found")
        return None
    return module_spec


def import_module_from_spec(module_spec):
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def op_implement_register(kind):
    def _register(cls):
        if kind not in OP_IMPLEMENTS:
            OP_IMPLEMENTS[kind] = {
                "cls": cls,
                "module": get_current_context().get_module(),
                "name": cls.__name__,
            }
        return cls

    return _register


def op_register(module):
    module_spec = check_module(module)
    if module_spec:
        with CustomOpContext(module):
            module = import_module_from_spec(module_spec)


OP_SHAPE_INFERS = {}


def op_shape_infer_register(kind):
    def _register(func):
        if kind not in OP_SHAPE_INFERS:

            def _infer_shape(inputs_shape):
                return func(inputs_shape)

            OP_SHAPE_INFERS[kind] = _infer_shape
        return func

    return _register


def get_op_shape_infer(kind):
    if kind in OP_SHAPE_INFERS:
        return OP_SHAPE_INFERS[kind]
    return None


def get_all_op_shape_infer():
    return OP_SHAPE_INFERS


def remove_custom_op_input(model: "OnnxModel") -> "OnnxModel":
    for node in model.graph.nodes:
        for node_input in node.inputs[:]:
            if node_input.name.endswith("_hz_input"):
                node.remove_input(node_input)
    return model


def check_custom_op_impl(model: "OnnxModel") -> "OnnxModel":
    def _find_impl(node):
        class_name = node.attributes.get("class_name", None)
        module = node.attributes.get("module", None)
        for impl in get_all_op_implements().values():
            if impl["name"] == class_name and impl["module"] == module:
                return impl
        return None

    pyop_nodes = model.graph.type2nodes["PyOp"]
    for node in pyop_nodes:
        if _find_impl(node) is None:
            # 如果用户没有调用注册函数, 直接返回注册失败.
            # 不再设置默认的op属性及输入情况, 以避免未注册/注册失败的情况未能及时发现.
            logging.error(
                f"Failed to load custom operator implenmentation: {node.name}"
            )
            raise ValueError("Implenmentation for custom op is not found")
    return model
