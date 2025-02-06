from numba import njit as hbdk4_njit  # noqa: F401
from hbdk4.compiler.overlay import Module


def translate(func, *inputs, **options) -> Module:
    """Translate the graph represented by func into Module

    Args:
        * func (function): callable python function that represents a compute graph, it may contain
                         numba jitted function, torch model, leap.custom op or Module from leap.call
        * inputs (np.ndarray/torch.tensor): parameters passed to func.

    Returns:
        * Module: a helper for mlir.Module that manages hbdk operations
    """
    # call the trace function
    from hbdk4.compiler.numba.trace import Translator

    translator = Translator(func, *inputs, **options)
    translator.translate()

    # NOTE-HACK-FOR-NUMBA-INTERPRETER
    m = Module(translator.module)
    from hbdk4.compiler.numba.tools import compile_numba

    m = compile_numba(m)
    return m
