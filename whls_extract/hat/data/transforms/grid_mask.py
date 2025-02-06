from typing import Any

import numpy as np
from PIL import Image

from hat.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class GridMask(object):
    """Generate GridMask for grid masking augmentation.

    Args:
        use_h (bool): if gen grid for height dim.
        use_w (bool): if gen grid for weight dim.
        rotate (float): Rotation of grid mesh.
        offset (bool): if randomly add offset.
        ratio (float): black grid mask ratio.
        limit_d_ratio_min (float): min black add white mask ratio.
        limit_d_ratio_max (float): max black add white mask ratio.
        mode (int): 0 or 1, if use ~mask.
        prob (float): probablity of occurance.
    """

    def __init__(
        self,
        use_h: bool,
        use_w: bool,
        rotate: float = 1.0,
        offset: bool = False,
        ratio: float = 0.5,
        limit_d_ratio_min: float = 0.0,
        limit_d_ratio_max: float = 1.0,
        mode: int = 0,
        prob: float = 1.0,
    ):
        super(GridMask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.limit_d_ratio_min = limit_d_ratio_min
        self.limit_d_ratio_max = limit_d_ratio_max
        self.mode = mode
        self.st_prob = prob
        self.prob = prob

    def set_prob(self, epoch: int, max_epoch: int) -> Any:
        self.prob = self.st_prob * epoch / max_epoch  # + 1.#0.5

    def __call__(self, data):
        if np.random.rand() > self.prob:
            return data

        if isinstance(data, dict):
            x = data["img"]
        else:
            x = data

        h, w, c = x.shape
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(
            max(2, h * self.limit_d_ratio_min), h * self.limit_d_ratio_max
        )
        self.l_ = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l_, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l_, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[
            (hh - h) // 2 : (hh - h) // 2 + h,
            (ww - w) // 2 : (ww - w) // 2 + w,
        ]

        if self.mode == 1:
            mask = 1 - mask

        mask = np.expand_dims(mask.astype(np.float32), axis=2)
        mask = np.tile(mask, [1, 1, c])

        if self.offset:
            offset = np.float32(2 * (np.random.rand(h, w) - 0.5))
            x = x * mask + offset * (1 - mask)
        else:
            x = x * mask

        if isinstance(data, dict):
            data["img"] = x
            return data
        else:
            return x
