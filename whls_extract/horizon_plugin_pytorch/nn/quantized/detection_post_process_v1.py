import math
from typing import Callable

import torch
from torch import Tensor, nn
from torch.jit.annotations import List, Optional, Tuple

from horizon_plugin_pytorch.dtype import qint8, qint16
from horizon_plugin_pytorch.nn import qat
from horizon_plugin_pytorch.qtensor import QTensor
from horizon_plugin_pytorch.utils.typeguard import typechecked
from .functional import detection_post_process_v1


class DetectionPostProcessV1(nn.Module):
    _QAT_MODULE = qat.DetectionPostProcessV1

    def __init__(
        self,
        num_classes: int,
        box_filter_threshold: float,
        class_offsets: List[int],
        use_clippings: List[bool],
        image_size: Tuple[int, int],
        nms_threshold: float,
        post_nms_top_k: int,
        nms_margin: float,
        use_stable_sort: bool = None,
        bbox_min_hw=None,
        exp_overwrite: Callable = torch.exp,
        input_shift: int = 4,
    ):
        super(DetectionPostProcessV1, self).__init__()
        self.num_classes = num_classes
        self.box_filter_threshold = box_filter_threshold
        self.class_offsets = class_offsets
        self.use_clippings = use_clippings
        self.image_size = torch.tensor(image_size).reshape(1, 2)
        self.nms_threshold = nms_threshold
        self.post_nms_top_k = post_nms_top_k
        self.nms_margin = nms_margin
        self.use_stable_sort = use_stable_sort
        self.exp_overwrite = exp_overwrite
        self.input_shift = input_shift
        assert bbox_min_hw is None

        self.kSRamSize = 4095
        self.kOutputBBoxShift = 2
        self.kNmsThresholdShift = 8

        self.register_buffer("in_scale", torch.ones(1, dtype=torch.float32))
        self.register_buffer("out_scale", torch.ones(1, dtype=torch.float32))
        self.register_buffer("exp_table", torch.ones(256, dtype=torch.int32))

        self.exp_shift = None
        self.seed = 1

    def _convert_float_params_to_int(self):
        self.box_filter_threshold = math.ceil(
            self.box_filter_threshold * (1 << self.input_shift)
        )

        self.nms_threshold = math.ceil(
            self.nms_threshold * (1 << self.kNmsThresholdShift)
        )
        self.nms_margin = math.ceil(self.nms_margin * (1 << self.input_shift))

    def _gen_exp_table(self):
        delta_scale = self.in_scale
        int_idx = torch.arange(
            -128, 128, dtype=torch.float, device=delta_scale.device
        )
        float_idx = int_idx * delta_scale[0]
        float_exp_value = self.exp_overwrite(float_idx)
        bit_num = torch.log2(float_exp_value.max() + 1).ceil().item()
        # exp_shift must be not smaller than input_shift
        self.exp_shift = int(max(15 - bit_num, self.input_shift))
        int_exp_value = (
            (float_exp_value * (1 << self.exp_shift))
            .round()
            .to(dtype=torch.int32)
        )
        self.exp_table.copy_(int_exp_value)

    @typechecked
    def forward(
        self,
        data: List[QTensor],
        anchor: List[Tensor],
        image_sizes: Optional[Tensor] = None,
    ) -> List[Tuple[QTensor, QTensor, Tensor]]:
        shifted_data = []
        for per_branch_data in data:
            if torch.equal(
                per_branch_data.q_scale(),
                self.in_scale.expand_as(per_branch_data.q_scale()),
            ):
                shifted_data.append(per_branch_data.int_repr())
            else:
                raise ValueError(
                    "DetectionPostProcessV1 requires"
                    " all inputs to be scale = 1 / 2**{}, "
                    "but receive scale = {}".format(
                        self.input_shift, per_branch_data.q_scale()[0].item()
                    )
                )

        ret = detection_post_process_v1(
            shifted_data,
            anchor,
            self.exp_table,
            image_sizes if image_sizes is not None else self.image_size,
            self.num_classes,
            [self.input_shift] * len(data),
            self.exp_shift,
            self.box_filter_threshold,
            self.class_offsets,
            self.seed,
            self.use_clippings,
            self.nms_threshold,
            self.nms_margin,
            self.post_nms_top_k,
            self.use_stable_sort,
        )

        return [
            (
                QTensor(r[0], self.out_scale, qint16),
                QTensor(r[1], self.in_scale, qint8),
                r[2],
            )
            for r in ret
        ]

    @classmethod
    def from_float(cls, mod: qat.DetectionPostProcessV1):
        r"""Create a quantized module from a qat module."""
        assert type(mod) == cls._QAT_MODULE, (
            "quantized."
            + cls.__name__
            + ".from_float only works for "
            + cls._QAT_MODULE.__name__
        )

        quantized_mod: DetectionPostProcessV1 = cls(
            num_classes=mod.num_classes,
            box_filter_threshold=mod.box_filter_threshold,
            class_offsets=mod.class_offsets,
            use_clippings=mod.use_clippings,
            image_size=mod.image_size,
            nms_threshold=mod.nms_threshold,
            post_nms_top_k=mod.post_nms_top_k,
            nms_margin=mod.nms_margin,
            use_stable_sort=mod.use_stable_sort,
            exp_overwrite=mod.exp_overwrite,
            input_shift=mod.input_shift,
        )

        quantized_mod.in_scale.copy_(mod.in_scale)
        quantized_mod.out_scale.copy_(mod.out_scale)
        quantized_mod._gen_exp_table()
        quantized_mod._convert_float_params_to_int()

        return quantized_mod
