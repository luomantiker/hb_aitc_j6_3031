from typing import Callable, Optional

import torch
from torch import Tensor
from torch.jit.annotations import List, Tuple

from horizon_plugin_pytorch.dtype import qinfo, qint8, qint16
from horizon_plugin_pytorch.nn.detection_post_process_v1 import (
    DetectionPostProcessV1 as FloatDetectionPostProcessV1,
)
from horizon_plugin_pytorch.nn.qat.qat_meta import (
    init_input_preprocess,
    pre_process,
)
from horizon_plugin_pytorch.qtensor import QTensor
from horizon_plugin_pytorch.utils.typeguard import typechecked
from .functional import scale_quanti


class DetectionPostProcessV1(FloatDetectionPostProcessV1):
    _FLOAT_MODULE = FloatDetectionPostProcessV1

    def __init__(
        self,
        num_classes: int,
        box_filter_threshold: float,
        class_offsets: List[int],
        use_clippings: List[bool],
        image_size: Tuple[int, int],
        nms_threshold: float,
        pre_nms_top_k: int,
        post_nms_top_k: int,
        qconfig,
        nms_padding_mode: Optional[str] = None,
        nms_margin: float = 0.0,
        use_stable_sort: bool = None,
        bbox_min_hw: Tuple[float, float] = (0, 0),
        exp_overwrite: Callable = torch.exp,
        input_shift: int = 4,
    ):
        super(DetectionPostProcessV1, self).__init__(
            num_classes,
            box_filter_threshold,
            class_offsets,
            use_clippings,
            image_size,
            nms_threshold,
            pre_nms_top_k,
            post_nms_top_k,
            nms_padding_mode=nms_padding_mode,
            nms_margin=nms_margin,
            use_stable_sort=use_stable_sort,
            bbox_min_hw=bbox_min_hw,
            exp_overwrite=exp_overwrite,
            input_shift=input_shift,
        )
        self.kOutputBBoxShift = 2
        self.kNmsThresholdShift = 8

        self.register_buffer(
            "in_scale",
            torch.ones(1, dtype=torch.float32) / (1 << self.input_shift),
        )
        self.register_buffer(
            "out_scale",
            torch.ones(1, dtype=torch.float32) / (1 << self.kOutputBBoxShift),
        )

        self.qconfig = qconfig
        self.activation_pre_process = init_input_preprocess(qconfig)

    @typechecked
    def forward(
        self,
        data: List[QTensor],
        anchors: List[Tensor],
        image_sizes=None,
    ) -> List[Tuple[QTensor, QTensor, Tensor]]:
        data = pre_process(self.activation_pre_process, data)
        shifted_data = []
        shifted_anchor = []
        for per_branch_data in data:
            if torch.equal(
                per_branch_data.q_scale(),
                self.in_scale.expand_as(per_branch_data.q_scale()),
            ):
                shifted_data.append(per_branch_data.as_subclass(torch.Tensor))
            else:
                raise ValueError(
                    "DetectionPostProcessV1 requires"
                    " all inputs to be scale = 1 / 2**{}, "
                    "but receive scale = {}".format(
                        self.input_shift, per_branch_data.q_scale()[0].item()
                    )
                )

        for per_branch_anchor in anchors:
            shifted_anchor.append(
                scale_quanti(
                    per_branch_anchor.to(dtype=torch.float),
                    self.out_scale,
                    torch.zeros_like(self.out_scale).to(dtype=torch.long),
                    -1,
                    -1 << 31,
                    (1 << 31) - 1,
                    True,
                    False,
                )
            )

        ret = super(DetectionPostProcessV1, self).forward(
            shifted_data, shifted_anchor, image_sizes
        )

        from horizon_plugin_pytorch.quantization import hbdk4 as hb4

        if hb4.is_exporting():
            # return single output during exporting to align to hbir output
            return [
                QTensor(
                    scale_quanti(
                        r.to(dtype=torch.float)[:, : self.post_nms_top_k, :],
                        self.out_scale,
                        torch.zeros_like(self.out_scale).to(dtype=torch.long),
                        -1,
                        qinfo(qint16).min,
                        qinfo(qint16).max,
                        True,
                        False,
                    ),
                    self.out_scale,
                    qint16,
                )
                for r in ret
            ]
        else:
            return [
                (
                    QTensor(
                        scale_quanti(
                            r[0].to(dtype=torch.float),
                            self.out_scale,
                            torch.zeros_like(self.out_scale).to(
                                dtype=torch.long
                            ),
                            -1,
                            qinfo(qint16).min,
                            qinfo(qint16).max,
                            True,
                            False,
                        ),
                        self.out_scale,
                        qint16,
                    ),
                    QTensor(
                        scale_quanti(
                            r[1].to(dtype=torch.float),
                            self.in_scale,
                            torch.zeros_like(self.in_scale).to(
                                dtype=torch.long
                            ),
                            -1,
                            qinfo(qint8).min,
                            qinfo(qint8).max,
                            True,
                            False,
                        ),
                        self.in_scale,
                        qint8,
                    ),
                    r[2],
                )
                for r in ret
            ]

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict.

        Args: `mod` a float module
        """
        assert type(mod) == cls._FLOAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
        )
        assert hasattr(
            mod, "qconfig"
        ), "Input float module must have qconfig defined"
        assert mod.qconfig, "Input float module must have a valid qconfig"

        qat_mod = cls(
            num_classes=mod.num_classes,
            box_filter_threshold=mod.box_filter_threshold,
            class_offsets=mod.class_offsets,
            use_clippings=mod.use_clippings,
            image_size=mod.image_size,
            nms_threshold=mod.nms_threshold,
            pre_nms_top_k=mod.pre_nms_top_k,
            post_nms_top_k=mod.post_nms_top_k,
            qconfig=mod.qconfig,
            nms_padding_mode=mod.nms_padding_mode,
            nms_margin=mod.nms_margin,
            use_stable_sort=mod.use_stable_sort,
            bbox_min_hw=mod.bbox_min_hw,
            exp_overwrite=mod.exp_overwrite,
            input_shift=mod.input_shift,
        )
        return qat_mod
