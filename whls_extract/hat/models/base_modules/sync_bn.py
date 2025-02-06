"""In torch>=1.12, `torch.nn.SyncBatchNorm` added below code.

https://github.com/pytorch/pytorch/blob/release/2.0/torch/nn/modules/_functions.py#L73-L85,
which will cause the training speed to be slower.

So, we implemented the SyncBatchNorm module based on `torch.nn.SyncBatchNorm`.
Compared with the original `torch.nn.SyncBatchNorm`, we only removed the code:
https://github.com/pytorch/pytorch/blob/release/2.0/torch/nn/modules/_functions.py#L73-L85,
and the other parts have not changed. And it will be faster in training.
"""

from typing import Any, Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.autograd.function import Function
from torch.nn import functional as F


class SyncBatchNormFunc(Function):
    @staticmethod
    def forward(
        self,
        input,
        weight,
        bias,
        running_mean,
        running_var,
        eps,
        momentum,
        process_group,
        world_size,
    ):
        if not (
            input.is_contiguous(memory_format=torch.channels_last)
            or input.is_contiguous(memory_format=torch.channels_last_3d)
        ):
            input = input.contiguous()
        if weight is not None:
            weight = weight.contiguous()

        size = int(input.numel() // input.size(1))
        if size == 1 and world_size < 2:
            raise ValueError(
                "Expected more than 1 value per channel when training, got input size {}".format(  # noqa E501
                    size
                )
            )

        num_channels = input.shape[1]
        if input.numel() > 0:
            # calculate mean/invstd for input.
            mean, invstd = torch.batch_norm_stats(input, eps)

            count = torch.full(
                (1,),
                input.numel() // input.size(1),
                dtype=mean.dtype,
                device=mean.device,
            )

            # C, C, 1 -> (2C + 1)
            combined = torch.cat([mean, invstd, count], dim=0)
        else:
            # for empty input, set stats and the count to zero. The stats with
            # zero count will be filtered out later when computing global mean
            # & invstd, but they still needs to participate the all_gather
            # collective communication to unblock other peer processes.
            combined = torch.zeros(
                2 * num_channels + 1, dtype=input.dtype, device=input.device
            )

        # Use allgather instead of allreduce because count could be different
        # across ranks, simple all reduce op can not give correct results.
        # batch_norm_gather_stats_with_counts calculates global mean & invstd
        # based on all gathered mean, invstd and count.
        # for nccl backend, use the optimized version of all gather.
        if process_group._get_backend_name() == "nccl":
            # world_size * (2C + 1)
            combined_size = combined.numel()
            combined_flat = torch.empty(
                1,
                combined_size * world_size,
                dtype=combined.dtype,
                device=combined.device,
            )
            dist.all_gather_into_tensor(
                combined_flat, combined, process_group, async_op=False
            )
            combined = torch.reshape(
                combined_flat, (world_size, combined_size)
            )
            # world_size * (2C + 1) -> world_size * C, world_size * C, world_size * 1   # noqa E501
            mean_all, invstd_all, count_all = torch.split(
                combined, num_channels, dim=1
            )
        else:
            # world_size * (2C + 1)
            combined_list = [
                torch.empty_like(combined) for _ in range(world_size)
            ]
            dist.all_gather(
                combined_list, combined, process_group, async_op=False
            )
            combined = torch.stack(combined_list, dim=0)
            # world_size * (2C + 1) -> world_size * C, world_size * C, world_size * 1  # noqa E501
            mean_all, invstd_all, count_all = torch.split(
                combined, num_channels, dim=1
            )

        # Note: The following code will cause the training speed to slow down
        # in `torch>=1.13.0`, so remove it.

        # if not torch.cuda.is_current_stream_capturing():
        #     # The lines below force a synchronization between CUDA and CPU,
        #     # because the shape of the result count_all depends on the
        #     # values in mask tensor.
        #     # Such synchronizations break CUDA Graph capturing.
        #     # See https://github.com/pytorch/pytorch/issues/78549
        #     # FIXME: https://github.com/pytorch/pytorch/issues/78656
        #     # describes a better longer-term solution.

        #     # remove stats from empty inputs
        #     mask = count_all.squeeze(-1) >= 1
        #     count_all = count_all[mask]
        #     mean_all = mean_all[mask]
        #     invstd_all = invstd_all[mask]

        # calculate global mean & invstd
        counts = count_all.view(-1)
        if running_mean is not None and counts.dtype != running_mean.dtype:
            counts = counts.to(running_mean.dtype)
        mean, invstd = torch.batch_norm_gather_stats_with_counts(
            input,
            mean_all,
            invstd_all,
            running_mean,
            running_var,
            momentum,
            eps,
            counts,
        )

        self.save_for_backward(
            input, weight, mean, invstd, count_all.to(torch.int32)
        )
        self.process_group = process_group

        # apply element-wise normalization
        if input.numel() > 0:
            return torch.batch_norm_elemt(
                input, weight, bias, mean, invstd, eps
            )
        else:
            return torch.empty_like(input)

    @staticmethod
    def backward(self, grad_output):
        if not (
            grad_output.is_contiguous(memory_format=torch.channels_last)
            or grad_output.is_contiguous(memory_format=torch.channels_last_3d)
        ):
            grad_output = grad_output.contiguous()
        saved_input, weight, mean, invstd, count_tensor = self.saved_tensors
        grad_input = grad_weight = grad_bias = None
        process_group = self.process_group

        if saved_input.numel() > 0:
            # calculate local stats as well as grad_weight / grad_bias
            (
                sum_dy,
                sum_dy_xmu,
                grad_weight,
                grad_bias,
            ) = torch.batch_norm_backward_reduce(
                grad_output,
                saved_input,
                mean,
                invstd,
                weight,
                self.needs_input_grad[0],
                self.needs_input_grad[1],
                self.needs_input_grad[2],
            )

            if self.needs_input_grad[0]:
                # synchronizing stats used to calculate input gradient.
                num_channels = sum_dy.shape[0]
                combined = torch.cat([sum_dy, sum_dy_xmu], dim=0)
                torch.distributed.all_reduce(
                    combined,
                    torch.distributed.ReduceOp.SUM,
                    process_group,
                    async_op=False,
                )
                sum_dy, sum_dy_xmu = torch.split(combined, num_channels)

                # backward pass for gradient calculation
                if weight is not None and weight.dtype != mean.dtype:
                    weight = weight.to(mean.dtype)
                grad_input = torch.batch_norm_backward_elemt(
                    grad_output,
                    saved_input,
                    mean,
                    invstd,
                    weight,
                    sum_dy,
                    sum_dy_xmu,
                    count_tensor,
                )
            # synchronizing of grad_weight / grad_bias is not needed as
            # distributed training would handle all reduce.
            if weight is None or not self.needs_input_grad[1]:
                grad_weight = None

            if weight is None or not self.needs_input_grad[2]:
                grad_bias = None
        else:
            # This process got an empty input tensor in the forward pass.
            # Although this process can directly set grad_input as an empty
            # tensor of zeros, it still needs to participate in the collective
            # communication to unblock its peers, as other peer processes might
            # have received non-empty inputs.
            num_channels = saved_input.shape[1]
            if self.needs_input_grad[0]:
                # launch all_reduce to unblock other peer processes
                combined = torch.zeros(
                    2 * num_channels,
                    dtype=saved_input.dtype,
                    device=saved_input.device,
                )
                torch.distributed.all_reduce(
                    combined,
                    torch.distributed.ReduceOp.SUM,
                    process_group,
                    async_op=False,
                )

            # Leave grad_input, grad_weight and grad_bias as None, which will
            # be interpreted by the autograd engine as Tensors full of zeros.

        return (
            grad_input,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class SyncBatchNorm(torch.nn.SyncBatchNorm):
    r"""Overwrite forward process of `torch.nn.SyncBatchNorm` to train faster.

    Note: Only the forward process is different from `torch.nn.SyncBatchNorm`,
    the rest of the modules are exactly the same as `torch.nn.SyncBatchNorm`.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, +)`
        eps: a value added to the denominator for numerical stability.
            Default: ``1e-5``
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to
            ``False``, this module does not track such statistics,
            and initializes statistics buffers :attr:`running_mean`
            and :attr:`running_var` as ``None``. When these buffers are
            ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``
        process_group: synchronization of stats happen within each process
            group individually. Default behavior is synchronization across the
            whole world.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        process_group: Optional[Any] = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats,
            process_group,
            device,
            dtype,
        )

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)
        self._check_non_zero_input_channels(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            assert self.num_batches_tracked is not None
            self.num_batches_tracked = self.num_batches_tracked + 1
            # self.num_batches_tracked.add_(1)
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = (
                    1.0 / self.num_batches_tracked.item()
                )
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization
        rather than the buffers. Mini-batch stats are used in training mode,
        and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (
                self.running_var is None
            )

        r"""
        Buffers are only updated if they are to be tracked and we are in
        training mode. Thus they only need to be passed when the update
        should occur (i.e. in training mode when they are tracked), or when
        buffer stats are used for normalization (i.e. in eval mode when
        buffers are not None).
        """
        # If buffers are not to be tracked, ensure that they won't be updated
        running_mean = (
            self.running_mean
            if not self.training or self.track_running_stats
            else None
        )
        running_var = (
            self.running_var
            if not self.training or self.track_running_stats
            else None
        )

        # Don't sync batchnorm stats in inference mode (model.eval()).
        need_sync = (
            bn_training
            and self.training
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        # need_sync = bn_training and self.training
        if need_sync:
            # currently only GPU input is supported
            if not input.is_cuda:
                raise ValueError(
                    "SyncBatchNorm expected input tensor to be on GPU"
                )

            process_group = torch.distributed.group.WORLD
            if self.process_group:
                process_group = self.process_group
            world_size = torch.distributed.get_world_size(process_group)
            need_sync = world_size > 1

        # fallback to framework BN when synchronization is not necessary
        if not need_sync:
            return F.batch_norm(
                input,
                running_mean,
                running_var,
                self.weight,
                self.bias,
                bn_training,
                exponential_average_factor,
                self.eps,
            )
        else:
            assert bn_training
            return SyncBatchNormFunc.apply(
                input,
                self.weight,
                self.bias,
                running_mean,
                running_var,
                self.eps,
                exponential_average_factor,
                process_group,
                world_size,
            )

    @classmethod
    def convert_sync_batchnorm(cls, module, process_group=None):
        r"""Overwrite convert_sync_batchnorm in `torch.nn.SyncBatchNorm`.

        Args:
            module: module containing one or more :attr:`BatchNorm*D` layers.
            process_group: process group to scope synchronization,
                default is the whole world

        Returns:
            The original :attr:`module` with the converted
             :class:`torch.nn.SyncBatchNorm` layers. If the original
             :attr:`module` is a :attr:`BatchNorm*D` layer, a new
             :class:`torch.nn.SyncBatchNorm` layer object will be returned
            instead.
        """
        module_output = module
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module_output = SyncBatchNorm(
                module.num_features,
                module.eps,
                module.momentum,
                module.affine,
                module.track_running_stats,
                process_group,
            )
            if module.affine:
                with torch.no_grad():
                    module_output.weight = module.weight
                    module_output.bias = module.bias
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked
            module_output.training = module.training
            if hasattr(module, "qconfig"):
                module_output.qconfig = module.qconfig
        for name, child in module.named_children():
            module_output.add_module(
                name, cls.convert_sync_batchnorm(child, process_group)
            )
        del module
        return module_output
