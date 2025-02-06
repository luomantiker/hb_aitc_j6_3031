from hmct.custom import op_registration

__all__ = ["op_implement_register", "op_register", "op_shape_infer_register"]


def op_register(module):
    """Register op module.

    Args:
        module (str): Module name.

    """
    op_registration.op_register(module)


def op_implement_register(kind):
    """Register op implement.

    Args:
        kind (str): Custom op type.

    ex.
        @op_implement_register("Sample")
        class SampleCustom(CustomModule):
            def __init__(self, test1, test2=2, test3=3):
                self._test1 = test1

            def forward(self, X):
                return X
    """
    return op_registration.op_implement_register(kind)


def op_shape_infer_register(kind):
    """Register op shape infer function.

    Args:
        kind (str): Custom op type.

    ex.
       @op_shape_infer_register("Sample")
       def infer_shape(input_shapes):
           return output_shapes
    """
    return op_registration.op_shape_infer_register(kind)
