import copy

import torch
from torch.nn import (
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose2d,
    Linear,
    Module,
    ReLU,
    ReLU6,
)
from torch.nn.quantized import FloatFunctional

from ..channel_scale import ChannelScale2d


class FusedConv1d(Conv1d):
    @classmethod
    def get_fuse_types(cls):
        return (Conv1d,)

    @classmethod
    def fuse_from(cls, *mods):
        for mod_type, mod in zip(cls.get_fuse_types(), mods):
            if not isinstance(mod_type, tuple):
                mod_type = (mod_type,)

            if type(mod) not in mod_type:
                raise ValueError(
                    "Expect type {}, but receive {}".format(
                        mod_type, type(mod)
                    )
                )

        conv = mods[0]
        # Do not inplaced modify conv, because conv maybe used
        # in other fuse patterns!
        fused = copy.deepcopy(conv)
        fused.__class__ = cls
        return fused


class ConvReLU1d(FusedConv1d):
    def forward(self, input):
        output = super().forward(input)
        return torch.nn.functional.relu(output)

    @classmethod
    def get_fuse_types(cls):
        return (Conv1d, ReLU)


class ConvReLU61d(FusedConv1d):
    def forward(self, input):
        output = super().forward(input)
        return torch.nn.functional.relu6(output)

    @classmethod
    def get_fuse_types(cls):
        return (Conv1d, ReLU6)


class ConvAdd1d(FusedConv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._swap_inputs = False

    def swap_inputs(self, v=True):
        self._swap_inputs = v

    def __call__(self, x, y):
        if self._swap_inputs:
            x, y = y, x
        return super().__call__(x, y)

    def add(self, x, y):
        return self.__call__(x, y)

    def forward(self, x, y):
        x = super().forward(x)
        return x + y

    @classmethod
    def get_fuse_types(cls):
        from horizon_plugin_pytorch.nn.quantized import (
            FloatFunctional as HFloatFunctional,
        )

        return (Conv1d, (FloatFunctional, HFloatFunctional))

    @classmethod
    def fuse_from(cls, *mods):
        fused = super().fuse_from(*mods)
        fused._swap_inputs = False
        return fused


class ConvAddReLU1d(ConvAdd1d):
    def forward(self, x, y):
        x = super().forward(x, y)
        return torch.nn.functional.relu(x)

    @classmethod
    def get_fuse_types(cls):
        from horizon_plugin_pytorch.nn.quantized import (
            FloatFunctional as HFloatFunctional,
        )

        return (Conv1d, (FloatFunctional, HFloatFunctional), ReLU)


class ConvAddReLU61d(ConvAdd1d):
    def forward(self, x, y):
        x = super().forward(x, y)
        return torch.nn.functional.relu6(x)

    @classmethod
    def get_fuse_types(cls):
        from horizon_plugin_pytorch.nn.quantized import (
            FloatFunctional as HFloatFunctional,
        )

        return (Conv1d, (FloatFunctional, HFloatFunctional), ReLU6)


class FusedConv2d(Conv2d):
    @classmethod
    def get_fuse_types(cls):
        return (Conv2d,)

    @classmethod
    def fuse_from(cls, *mods):
        for mod_type, mod in zip(cls.get_fuse_types(), mods):
            if not isinstance(mod_type, tuple):
                mod_type = (mod_type,)

            if type(mod) not in mod_type:
                raise ValueError(
                    "Expect type {}, but receive {}".format(
                        mod_type, type(mod)
                    )
                )

        conv = mods[0]
        # Do not inplaced modify conv, because conv maybe used
        # in other fuse patterns!
        fused = copy.deepcopy(conv)
        fused.__class__ = cls
        return fused


class ConvReLU2d(FusedConv2d):
    def forward(self, input):
        from horizon_plugin_pytorch.nn.qat.compatible_ops import relu

        output = super().forward(input)
        return relu(output, getattr(self, "use_relu6", False))

    @classmethod
    def get_fuse_types(cls):
        return (Conv2d, ReLU)


class ConvReLU62d(FusedConv2d):
    def forward(self, input):
        from horizon_plugin_pytorch.nn.qat.compatible_ops import relu6

        output = super().forward(input)
        return relu6(output)

    @classmethod
    def get_fuse_types(cls):
        return (Conv2d, ReLU6)


class ConvAdd2d(FusedConv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._swap_inputs = False

    def swap_inputs(self, v=True):
        self._swap_inputs = v

    def __call__(self, x, y):
        if self._swap_inputs:
            x, y = y, x
        return super().__call__(x, y)

    def add(self, x, y):
        return self.__call__(x, y)

    def forward(self, x, y):
        x = super().forward(x)
        return x + y

    @classmethod
    def get_fuse_types(cls):
        from horizon_plugin_pytorch.nn.quantized import (
            FloatFunctional as HFloatFunctional,
        )

        return (Conv2d, (FloatFunctional, HFloatFunctional))

    @classmethod
    def fuse_from(cls, *mods):
        fused = super().fuse_from(*mods)
        fused._swap_inputs = False
        return fused


class ConvAddReLU2d(ConvAdd2d):
    def forward(self, x, y):
        from horizon_plugin_pytorch.nn.qat.compatible_ops import relu

        x = super().forward(x, y)
        return relu(x, getattr(self, "use_relu6", False))

    @classmethod
    def get_fuse_types(cls):
        from horizon_plugin_pytorch.nn.quantized import (
            FloatFunctional as HFloatFunctional,
        )

        return (Conv2d, (FloatFunctional, HFloatFunctional), ReLU)


class ConvAddReLU62d(ConvAdd2d):
    def forward(self, x, y):
        from horizon_plugin_pytorch.nn.qat.compatible_ops import relu6

        x = super().forward(x, y)
        return relu6(x)

    @classmethod
    def get_fuse_types(cls):
        from horizon_plugin_pytorch.nn.quantized import (
            FloatFunctional as HFloatFunctional,
        )

        return (Conv2d, (FloatFunctional, HFloatFunctional), ReLU6)


class ConvTransposeReLU2d(Module):
    r"""Call the ConvTranspose2d and relu modules.

    During quantization this will be replaced with the corresponding fused
    module.
    """

    def __init__(self, conv_transpose2d, relu):
        super(ConvTransposeReLU2d, self).__init__()
        assert (
            type(conv_transpose2d) == ConvTranspose2d and type(relu) == ReLU
        ), "Incorrect types for input modules{}{}".format(
            type(conv_transpose2d), type(relu)
        )
        self.conv_transpose2d = conv_transpose2d
        self.relu = relu

    def forward(self, x1):
        x1 = self.conv_transpose2d(x1)
        return self.relu(x1)


class ConvTransposeAdd2d(Module):
    r"""This is a container which calls the ConvTranspose2d and add modules.

    During quantization this will be replaced with the corresponding fused
    module.
    """

    def __init__(self, conv_transpose2d, add):
        super(ConvTransposeAdd2d, self).__init__()
        from ..quantized import FloatFunctional as HFloatFunctional

        assert type(conv_transpose2d) == ConvTranspose2d and (
            type(add) == FloatFunctional or type(add) == HFloatFunctional
        ), "Incorrect types for input modules{}{}".format(
            type(conv_transpose2d), type(add)
        )
        self.conv_transpose2d = conv_transpose2d
        self._swap_inputs = False

    def swap_inputs(self, v=True):
        self._swap_inputs = v

    def __call__(self, x, y):
        if self._swap_inputs:
            x, y = y, x
        return super().__call__(x, y)

    def add(self, x1, x2):
        return self.__call__(x1, x2)

    def forward(self, x1, x2):
        x1 = self.conv_transpose2d(x1)
        return x1 + x2


class ConvTransposeAddReLU2d(ConvTransposeAdd2d):
    r"""Calls the ConvTranspose2d and add relu modules.

    During quantization this will be replaced with the corresponding
    fused module.
    """

    def __init__(self, conv_transpose2d, add, relu):
        from ..quantized import FloatFunctional as HFloatFunctional

        assert (
            type(conv_transpose2d) == ConvTranspose2d
            and (type(add) == FloatFunctional or type(add) == HFloatFunctional)
            and type(relu) == ReLU
        ), "Incorrect types for input modules{}{}{}".format(
            type(conv_transpose2d), type(add), type(relu)
        )
        super(ConvTransposeAddReLU2d, self).__init__(conv_transpose2d, add)
        self.relu = relu

    def forward(self, x1, x2):
        out = super().forward(x1, x2)
        return self.relu(out)


class ConvTransposeReLU62d(Module):
    r"""Call the ConvTranspose2d and ReLU6 modules.

    During quantization this will be replaced with the corresponding fused
    module.
    """

    def __init__(self, conv_transpose2d, relu6):
        super(ConvTransposeReLU62d, self).__init__()
        assert (
            type(conv_transpose2d) == ConvTranspose2d and type(relu6) == ReLU6
        ), "Incorrect types for input modules{}{}".format(
            type(conv_transpose2d), type(relu6)
        )
        self.conv_transpose2d = conv_transpose2d
        self.relu6 = relu6

    def forward(self, x):
        x = self.conv_transpose2d(x)
        return self.relu6(x)


class ConvTransposeAddReLU62d(ConvTransposeAdd2d):
    r"""Call the ConvTranspose2d and add relu6 modules.

    During quantization this will be replaced with the corresponding
    fused module.
    """

    def __init__(self, conv_transpose2d, add, relu6):
        from ..quantized import FloatFunctional as HFloatFunctional

        assert (
            type(conv_transpose2d) == ConvTranspose2d
            and (type(add) == FloatFunctional or type(add) == HFloatFunctional)
            and type(relu6) == ReLU6
        ), "Incorrect types for input modules{}{}{}".format(
            type(conv_transpose2d), type(add), type(relu6)
        )
        super(ConvTransposeAddReLU62d, self).__init__(conv_transpose2d, add)
        self.relu6 = relu6

    def forward(self, x1, x2):
        out = super().forward(x1, x2)
        return self.relu6(out)


class ConvAdd3d(Module):
    r"""Call the Conv3d and add modules.

    During quantization this will be replaced with the corresponding fused
    module.
    """

    def __init__(self, conv, add):
        super(ConvAdd3d, self).__init__()
        from ..quantized import FloatFunctional as HFloatFunctional

        assert type(conv) == Conv3d and (
            type(add) == FloatFunctional or type(add) == HFloatFunctional
        ), "Incorrect types for input modules{}{}".format(
            type(conv), type(add)
        )
        self.conv = conv
        self._swap_inputs = False

    def swap_inputs(self, v=True):
        self._swap_inputs = v

    def __call__(self, x, y):
        if self._swap_inputs:
            x, y = y, x
        return super().__call__(x, y)

    def add(self, x1, x2):
        return self.__call__(x1, x2)

    def forward(self, x1, x2):
        x1 = self.conv(x1)
        return x1 + x2


class ConvAddReLU3d(ConvAdd3d):
    r"""Call the Conv3d and add relu modules.

    During quantization this will be replaced with the corresponding fused
    module.
    """

    def __init__(self, conv, add, relu):
        from ..quantized import FloatFunctional as HFloatFunctional

        assert (
            type(conv) == Conv3d
            and (type(add) == FloatFunctional or type(add) == HFloatFunctional)
            and type(relu) == ReLU
        ), "Incorrect types for input modules{}{}{}".format(
            type(conv), type(add), type(relu)
        )
        super(ConvAddReLU3d, self).__init__(conv, add)

        self.relu = relu

    def forward(self, x1, x2):
        out = super().forward(x1, x2)
        return self.relu(out)


class ConvReLU63d(Module):
    r"""Call the Conv3d and ReLU6 modules.

    During quantization this will be replaced with the corresponding fused
    module.
    """

    def __init__(self, conv, relu6):
        super(ConvReLU63d, self).__init__()
        assert (
            type(conv) == Conv3d and type(relu6) == ReLU6
        ), "Incorrect types for input modules{}{}".format(
            type(conv), type(relu6)
        )
        self.conv = conv
        self.relu6 = relu6

    def forward(self, x):
        x1 = self.conv(x)
        return self.relu6(x1)


class ConvAddReLU63d(ConvAdd3d):
    r"""Call the Conv3d and add ReLU6 modules.

    During quantization this will be replaced with the corresponding fused
    module.
    """

    def __init__(self, conv, add, relu6):
        from ..quantized import FloatFunctional as HFloatFunctional

        assert (
            type(conv) == Conv3d
            and (type(add) == FloatFunctional or type(add) == HFloatFunctional)
            and type(relu6) == ReLU6
        ), "Incorrect types for input modules{}{}{}".format(
            type(conv), type(add), type(relu6)
        )

        super(ConvAddReLU63d, self).__init__(conv, add)
        self.relu6 = relu6

    def forward(self, x1, x2):
        out = super().forward(x1, x2)
        return self.relu6(out)


# with bn
class ConvBN2d(Module):
    r"""Call the Conv2d and BatchNorm2d modules.

    During quantization this will be replaced with the corresponding fused
    module.
    """

    def __init__(self, conv, bn):
        super(ConvBN2d, self).__init__()

        assert type(conv) == Conv2d and isinstance(
            bn, (torch.nn.modules.batchnorm._BatchNorm, ChannelScale2d)
        ), "Incorrect types for input modules{}{}".format(type(conv), type(bn))
        self.conv = conv
        self.bn = bn

    def forward(self, x1):
        x1 = self.conv(x1)
        return self.bn(x1)


class ConvBNAdd2d(Module):
    r"""Call the Conv2d, BatchNormand add modules.

    During quantization this will be replaced with the corresponding fused
    module.
    """

    def __init__(self, conv, bn, add):
        super(ConvBNAdd2d, self).__init__()
        from ..quantized import FloatFunctional as HFloatFunctional

        assert (
            type(conv) == Conv2d
            and (type(add) == FloatFunctional or type(add) == HFloatFunctional)
            and isinstance(
                bn, (torch.nn.modules.batchnorm._BatchNorm, ChannelScale2d)
            )
        ), "Incorrect types for input modules{}{}{}".format(
            type(conv), type(add), type(bn)
        )
        self.conv = conv
        self.bn = bn
        self._swap_inputs = False

    def swap_inputs(self, v=True):
        self._swap_inputs = v

    def __call__(self, x, y):
        if self._swap_inputs:
            x, y = y, x
        return super().__call__(x, y)

    def add(self, x1, x2):
        return self.__call__(x1, x2)

    def forward(self, x1, x2):
        x1 = self.conv(x1)
        x1 = self.bn(x1)
        return x1 + x2


class ConvBNAddReLU2d(ConvBNAdd2d):
    r"""Call the Conv2d, BatchNorm and add relu modules.

    During quantization this will be replaced with the corresponding
    fused module.
    """

    def __init__(self, conv, bn, add, relu):
        from ..quantized import FloatFunctional as HFloatFunctional

        assert (
            type(conv) == Conv2d
            and isinstance(
                bn, (torch.nn.modules.batchnorm._BatchNorm, ChannelScale2d)
            )
            and (type(add) == FloatFunctional or type(add) == HFloatFunctional)
            and type(relu) == ReLU
        ), "Incorrect types for input modules{}{}{}{}".format(
            type(conv), type(bn), type(add), type(relu)
        )
        super(ConvBNAddReLU2d, self).__init__(conv, bn, add)
        self.relu = relu

    def forward(self, x1, x2):
        out = super().forward(x1, x2)
        return self.relu(out)


class ConvBNReLU2d(Module):
    r"""Call the Conv2d, BatchNorm2d and ReLU.

    During quantization this will be replaced with the corresponding
    fused module.
    """

    def __init__(self, conv, bn, relu):
        super(ConvBNReLU2d, self).__init__()
        assert (
            type(conv) == Conv2d
            and isinstance(
                bn, (torch.nn.modules.batchnorm._BatchNorm, ChannelScale2d)
            )
            and type(relu) == ReLU
        ), "Incorrect types for input modules{}{}{}".format(
            type(conv), type(bn), type(relu)
        )
        self.conv = conv
        self.relu = relu
        self.bn = bn

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class ConvBNReLU62d(Module):
    r"""Call the Conv2d, BatchNorm2d and ReLU6 modules.

    During quantization this will be replaced with the corresponding
    fused module.
    """

    def __init__(self, conv, bn, relu6):
        super(ConvBNReLU62d, self).__init__()
        assert (
            type(conv) == Conv2d
            and isinstance(
                bn, (torch.nn.modules.batchnorm._BatchNorm, ChannelScale2d)
            )
            and type(relu6) == ReLU6
        ), "Incorrect types for input modules{}{}{}".format(
            type(conv), type(bn), type(relu6)
        )
        self.conv = conv
        self.relu6 = relu6
        self.bn = bn

    def forward(self, x):
        x1 = self.conv(x)
        x1 = self.bn(x1)
        return self.relu6(x1)


class ConvBNAddReLU62d(ConvBNAdd2d):
    r"""Call the Conv2d and add ReLU6 modules.

    During quantization this will be replaced with the corresponding fused
    module.
    """

    def __init__(self, conv, bn, add, relu6):
        from ..quantized import FloatFunctional as HFloatFunctional

        assert (
            type(conv) == Conv2d
            and isinstance(
                bn, (torch.nn.modules.batchnorm._BatchNorm, ChannelScale2d)
            )
            and (type(add) == FloatFunctional or type(add) == HFloatFunctional)
            and type(relu6) == ReLU6
        ), "Incorrect types for input modules{}{}{}{}".format(
            type(conv), type(bn), type(add), type(relu6)
        )
        super(ConvBNAddReLU62d, self).__init__(conv, bn, add)

        self.relu6 = relu6

    def forward(self, x1, x2):
        out = super().forward(x1, x2)
        return self.relu6(out)


class LinearAdd(Module):
    r"""Call the Linear and add modules.

    During quantization this will be replaced with the corresponding fused
    module.
    """

    def __init__(self, linear, add):
        super(LinearAdd, self).__init__()
        from ..quantized import FloatFunctional as HFloatFunctional

        assert type(linear) == Linear and (
            type(add) == FloatFunctional or type(add) == HFloatFunctional
        ), "Incorrect types for input modules{}{}".format(
            type(linear), type(add)
        )
        self.linear = linear
        self._swap_inputs = False

    def swap_inputs(self, v=True):
        self._swap_inputs = v

    def __call__(self, x, y):
        if self._swap_inputs:
            x, y = y, x
        return super().__call__(x, y)

    def add(self, x1, x2):
        return self.__call__(x1, x2)

    def forward(self, x1, x2):
        x1 = self.linear(x1)
        return x1 + x2


class LinearAddReLU(LinearAdd):
    r"""Call the Linear and add and relu modules.

    During quantization this will be replaced with the corresponding fused
    module.
    """

    def __init__(self, linear, add, relu):
        from ..quantized import FloatFunctional as HFloatFunctional

        assert (
            type(linear) == Linear
            and (type(add) == FloatFunctional or type(add) == HFloatFunctional)
            and type(relu) == ReLU
        ), "Incorrect types for input modules{}{}{}".format(
            type(linear), type(add), type(relu)
        )
        super(LinearAddReLU, self).__init__(linear, add)
        self.relu = relu

    def forward(self, x1, x2):
        x = super().forward(x1, x2)
        return self.relu(x)


class LinearReLU(Module):
    r"""Call the Linear and ReLU modules.

    During quantization this will be replaced with the corresponding fused
    module.
    """

    def __init__(self, linear, relu):
        super(LinearReLU, self).__init__()
        assert (
            type(linear) == Linear and type(relu) == ReLU
        ), "Incorrect types for input modules{}{}".format(
            type(linear), type(relu)
        )
        self.linear = linear
        self.relu = relu

    def forward(self, x):
        x = self.linear(x)
        return self.relu(x)


class LinearReLU6(Module):
    r"""Calls the Linear and ReLU6 modules.

    During quantization this will be replaced with the corresponding fused
    module.
    """

    def __init__(self, linear, relu6):
        super(LinearReLU6, self).__init__()
        assert (
            type(linear) == Linear and type(relu6) == ReLU6
        ), "Incorrect types for input modules{}{}".format(
            type(linear), type(relu6)
        )
        self.linear = linear
        self.relu6 = relu6

    def forward(self, x):
        x = self.linear(x)
        return self.relu6(x)


class LinearAddReLU6(LinearAdd):
    r"""call the Linear and add and ReLU6 modules.

    During quantization this will be replaced with the corresponding fused
    module.
    """

    def __init__(self, linear, add, relu6):
        from ..quantized import FloatFunctional as HFloatFunctional

        assert (
            type(linear) == Linear
            and (type(add) == FloatFunctional or type(add) == HFloatFunctional)
            and type(relu6) == ReLU6
        ), "Incorrect types for input modules{}{}{}".format(
            type(linear), type(add), type(relu6)
        )
        super(LinearAddReLU6, self).__init__(linear, add)
        self.relu6 = relu6

    def forward(self, x1, x2):
        x = super().forward(x1, x2)
        return self.relu6(x)


class DeformConvReLU2d(Module):
    def __init__(self, deform_conv2d, relu):
        super(DeformConvReLU2d, self).__init__()
        self.deform_conv2d = deform_conv2d
        self.relu = relu

    def forward(self, input, offset, mask=None):
        out = self.deform_conv2d(input, offset, mask)
        return self.relu(out)


class DeformConvReLU62d(Module):
    def __init__(self, deform_conv2d, relu6):
        super(DeformConvReLU62d, self).__init__()
        self.deform_conv2d = deform_conv2d
        self.relu6 = relu6

    def forward(self, input, offset, mask=None):
        out = self.deform_conv2d(input, offset, mask)
        return self.relu6(out)


class DeformConvAdd2d(Module):
    def __init__(self, deform_conv2d, add):
        super(DeformConvAdd2d, self).__init__()
        self.deform_conv2d = deform_conv2d
        self.add_ff = add
        self._swap_inputs = False

    def swap_inputs(self, v=True):
        self._swap_inputs = v

    def __call__(self, x, y):
        if self._swap_inputs:
            x, y = y, x
        return super().__call__(x, y)

    def add(self, x1, x2):
        return self.__call__(x1, x2)

    def forward(self, x1, x2):
        out = self.deform_conv2d(*x1)
        return self.add_ff.add(out, x2)


class DeformConvAddReLU2d(DeformConvAdd2d):
    def __init__(self, deform_conv2d, add, relu):
        super(DeformConvAddReLU2d, self).__init__(deform_conv2d, add)
        self.relu = relu

    def forward(self, x1, x2):
        out = super().forward(x1, x2)
        return self.relu(out)


class DeformConvAddReLU62d(DeformConvAdd2d):
    def __init__(self, deform_conv2d, add, relu6):
        super(DeformConvAddReLU62d, self).__init__(deform_conv2d, add)
        self.relu6 = relu6

    def forward(self, x1, x2):
        out = super().forward(x1, x2)
        return self.relu6(out)
