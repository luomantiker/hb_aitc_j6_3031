"""This is a prototype feature."""
from itertools import permutations

import torch

from horizon_plugin_pytorch.utils.typeguard import typechecked


class MaskGenerator(object):
    @typechecked
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Generate mask. This is a prototype feature.

        Args:
            tensor (torch.Tensor): tensor for mask generation.

        Returns:
            torch.Tensor: mask tensor.
        """
        raise NotImplementedError

    @typechecked
    def is_maskable(self, tensor: torch.Tensor) -> bool:
        """
        Judge if a tensor is maskable. This is a prototype feature.

        Args:
            tensor (torch.Tensor): tensor for judgement.

        Returns:
            bool: if the tensor is maskable.
        """
        raise NotImplementedError


class SemistructedMaskGenerator(MaskGenerator):
    @typechecked
    def __init__(self, m: int = 4, n: int = 2):
        """Semi structed mask generator. This is a prototype feature.

        In every m elements, n elements are kept. Their mask values are set to
        one while mask values of other removal elements are set to zero.

        Args:
            m (int, optional): Group element num. Defaults to 4.
            n (int, optional): keep element num. Defaults to 2.
        """
        self.m = m
        self.n = n
        permutation = [1 if i < self.n else 0 for i in range(self.m)]
        self.mask_candidates = torch.FloatTensor(
            list(permutations(permutation)),
        )

    @typechecked
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Generate mask. This is a prototype feature.

        Args:
            tensor (torch.Tensor): tensor for mask generation.

        Returns:
            torch.Tensor: mask tensor.
        """
        shape = tensor.shape
        assert (
            0 < len(shape) < 5
        ), f"Can't generate mask for tensor with {len(shape)} dims"

        ttype = tensor.type()
        t = tensor.float().contiguous()

        if len(shape) == 1:
            t = t.view(1, shape[0])
            mask = self._unstructed_mn(t)
        elif len(shape) == 2:
            # linear
            t = t.view(shape[0], shape[1])
            mask = self._unstructed_mn(t)
        elif len(shape) == 3:
            # conv1d
            t = t.permute(0, 2, 1).contiguous().view(-1, shape[1])
            mask = self._unstructed_mn(t)
            mask = (
                mask.view(shape[0], shape[2], shape[1])
                .permute(0, 2, 1)
                .contiguous()
            )
        elif len(shape) == 4:
            # conv2d
            t = t.permute(2, 3, 0, 1).contiguous().view(-1, shape[1])
            mask = self._unstructed_mn(t)
            mask = (
                mask.view(shape[2], shape[3], shape[0], shape[1])
                .permute(2, 3, 0, 1)
                .contiguous()
            )

        return mask.view(shape).type(ttype)

    def _unstructed_mn(self, mat):
        origin_shape = mat.shape
        # (h, w) -> (hw / m, m)
        mat = mat.view(-1, self.m)
        mask = torch.ones_like(mat, device=mat.device)
        self.mask_candidates = self.mask_candidates.to(mat.device)
        pmax = torch.argmax(
            torch.matmul(mat.abs(), self.mask_candidates.t()),
            dim=1,
        )
        mask[:] = self.mask_candidates[pmax[:]]
        return mask.view(origin_shape)

    @typechecked
    def is_maskable(self, tensor: torch.Tensor) -> bool:
        """
        Judge if a tensor is maskable. This is a prototype feature.

        Args:
            tensor (torch.Tensor): tensor for judgement.

        Returns:
            bool: if the tensor is maskable.
        """
        if tensor.dim() == 1 and tensor.shape[0] % self.m == 0:
            return True
        elif 1 < tensor.dim() < 5 and tensor.shape[1] % self.m == 0:
            if tensor.dim() == 3 and tensor.shape[2] == 1:
                # conv1d
                return False
            elif (
                tensor.dim() == 4
                and tensor.shape[2] == 1
                and tensor.shape[3] == 1
            ):
                # conv2d
                return False
            return True
        else:
            return False


class UnstructedMaskGenerator(MaskGenerator):
    """Unstructed mask generator. This is a prototype feature.

    (prune_rate * element_num) elements are pruned. Their mask values are
    set to zero while mask values of other kept elements are set to one.
    """

    @typechecked
    def __call__(
        self,
        tensor: torch.Tensor,
        prune_rate: float = 0.75,
    ) -> torch.Tensor:
        """
        Generate mask. This is a prototype feature.

        Args:
            tensor (torch.Tensor): tensor for mask generation.
            prune_rate (float, optional): Zero element rate. Defaults to 0.75.

        Returns:
            torch.Tensor: mask tensor.
        """
        assert 0 <= prune_rate <= 1, "prune_rate should be in the range [0, 1]"
        shape = tensor.shape
        ttype = tensor.type()
        t = tensor.float().contiguous().flatten()
        mask = torch.ones_like(t, device=t.device)
        prune_element_num = int(t.numel() * prune_rate)
        prune_idx = t.abs().argsort()[:prune_element_num]
        mask[prune_idx] = 0

        return mask.view(shape).type(ttype)

    @typechecked
    def is_maskable(self, tensor: torch.Tensor) -> bool:
        """
        Judge if a tensor is maskable. This is a prototype feature.

        Args:
            tensor (torch.Tensor): tensor for judgement.

        Returns:
            bool: if the tensor is maskable.
        """
        return True
