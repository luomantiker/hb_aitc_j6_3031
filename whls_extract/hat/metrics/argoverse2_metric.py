# Copyright (c) Horizon Robotics. All rights reserved.
#
# Portions of this code are based on code by Zikang Zhou, used under the
# Apache License, Version 2.0.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
from typing import Optional, Tuple, Union

import numpy as np
import torch

from hat.metrics.metric import EvalMetric
from hat.registry import OBJECT_REGISTRY

__all__ = ["BrierMetric", "MinADE", "MinFDE", "MissRate", "HitRate"]


def wrap_angle(
    angle: Union[torch.Tensor, np.ndarray],
    min_val: float = -math.pi,
    max_val: float = math.pi,
) -> torch.Tensor:
    """Convert angels into [min_val, max_val].

    Args:
        angle: The angle vector.
        min_val: The minimum value of angles.
        max_val: The maximum value of angles.

    Returns:
        angles: Converted angles.
    """
    return min_val + (angle + max_val) % (max_val - min_val)


def topk(
    max_guesses: int,
    pred: torch.Tensor,
    prob: Optional[torch.Tensor] = None,
    ptr: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_guesses = min(max_guesses, pred.size(1))
    if max_guesses == pred.size(1):
        if prob is not None:
            prob = prob / prob.sum(dim=-1, keepdim=True)
        else:
            prob = pred.new_ones((pred.size(0), max_guesses)) / max_guesses
        return pred, prob
    else:
        if prob is not None:
            inds_topk = torch.topk(
                prob, k=max_guesses, dim=-1, largest=True, sorted=True
            )[1]
            pred_topk = pred[
                torch.arange(pred.size(0))
                .unsqueeze(-1)
                .expand(-1, max_guesses),
                inds_topk,
            ]
            prob_topk = prob[
                torch.arange(pred.size(0))
                .unsqueeze(-1)
                .expand(-1, max_guesses),
                inds_topk,
            ]
            prob_topk = prob_topk / prob_topk.sum(dim=-1, keepdim=True)
        else:
            pred_topk = pred[:, :max_guesses]
            prob_topk = (
                pred.new_ones((pred.size(0), max_guesses)) / max_guesses
            )
        return pred_topk, prob_topk


def valid_filter(
    pred: torch.Tensor,
    target: torch.Tensor,
    prob: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    keep_invalid_final_step: bool = True,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    if valid_mask is None:
        valid_mask = target.new_ones(target.size()[:-1], dtype=torch.bool)
    if keep_invalid_final_step:
        filter_mask = valid_mask.any(dim=-1)
    else:
        filter_mask = valid_mask[:, -1]
    pred = pred[filter_mask]
    target = target[filter_mask]
    if prob is not None:
        prob = prob[filter_mask]
    valid_mask = valid_mask[filter_mask]
    ptr = target.new_tensor([0, target.size(0)])
    return pred, target, prob, valid_mask, ptr


@OBJECT_REGISTRY.register
class BrierMetric(EvalMetric):
    """
    Brier metric for measuring the accuracy of probabilistic predictions.

    The Brier score measures the mean squared difference between predicted
    probabilities and the actual outcomes.
    Lower Brier scores indicate better predictive accuracy.

    Args:
        max_guesses: Maximum number of guesses to consider for
                top-k predictions.
        name: Name of the metric.
    """

    def __init__(
        self,
        max_guesses: int = 6,
        name: str = "Brier",
    ) -> None:
        super(BrierMetric, self).__init__(name)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.max_guesses = max_guesses

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        prob: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        keep_invalid_final_step: bool = True,
        min_criterion: str = "FDE",
    ) -> None:
        pred, target, prob, valid_mask, _ = valid_filter(
            pred, target, prob, valid_mask, keep_invalid_final_step
        )
        pred_topk, prob_topk = topk(self.max_guesses, pred, prob)
        if min_criterion == "FDE":
            inds_last = (
                valid_mask
                * torch.arange(1, valid_mask.size(-1) + 1, device=pred.device)
            ).argmax(dim=-1)
            inds_best = torch.norm(
                pred_topk[torch.arange(pred.size(0)), :, inds_last]
                - target[torch.arange(pred.size(0)), inds_last].unsqueeze(-2),
                p=2,
                dim=-1,
            ).argmin(dim=-1)
        elif min_criterion == "ADE":
            inds_best = (
                (
                    torch.norm(pred_topk - target.unsqueeze(1), p=2, dim=-1)
                    * valid_mask.unsqueeze(1)
                )
                .sum(dim=-1)
                .argmin(dim=-1)
            )
        else:
            raise ValueError(
                "{} is not a valid criterion".format(min_criterion)
            )
        self.sum += (
            (1.0 - prob_topk[torch.arange(pred.size(0)), inds_best])
            .pow(2)
            .sum()
        )
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count


@OBJECT_REGISTRY.register
class MinADE(EvalMetric):
    """
    Minimum Average Displacement Error (minADE) for trajectory predictions.

    The minADE calculates the average displacement error of the best
    prediction out of a set of predictions.

    Args:
        max_guesses: Maximum number of guesses to consider for
                top-k predictions.
        name: Name of the metric.

    """

    def __init__(self, max_guesses: int = 6, name="minADE") -> None:
        super(MinADE, self).__init__(name)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.max_guesses = max_guesses

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        prob: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        keep_invalid_final_step: bool = True,
        min_criterion: str = "FDE",
    ) -> None:
        pred, target, prob, valid_mask, _ = valid_filter(
            pred, target, prob, valid_mask, keep_invalid_final_step
        )
        pred_topk, _ = topk(self.max_guesses, pred, prob)
        if min_criterion == "FDE":
            inds_last = (
                valid_mask
                * torch.arange(1, valid_mask.size(-1) + 1, device=pred.device)
            ).argmax(dim=-1)
            inds_best = torch.norm(
                pred_topk[torch.arange(pred.size(0)), :, inds_last]
                - target[torch.arange(pred.size(0)), inds_last].unsqueeze(-2),
                p=2,
                dim=-1,
            ).argmin(dim=-1)
            self.sum += (
                (
                    torch.norm(
                        pred_topk[torch.arange(pred.size(0)), inds_best]
                        - target,
                        p=2,
                        dim=-1,
                    )
                    * valid_mask
                ).sum(dim=-1)
                / valid_mask.sum(dim=-1)
            ).sum()
        elif min_criterion == "ADE":
            self.sum += (
                (
                    torch.norm(pred_topk - target.unsqueeze(1), p=2, dim=-1)
                    * valid_mask.unsqueeze(1)
                )
                .sum(dim=-1)
                .min(dim=-1)[0]
                / valid_mask.sum(dim=-1)
            ).sum()
        else:
            raise ValueError(
                "{} is not a valid criterion".format(min_criterion)
            )
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count


@OBJECT_REGISTRY.register
class MinFDE(EvalMetric):
    """
    Minimum Final Displacement Error (minFDE) for trajectory predictions.

    The minFDE calculates the final displacement error of the best
    prediction out of a set of predictions.

    Args:
        max_guesses: Maximum number of guesses to consider
            for top-k predictions.
        name: Name of the metric.
    """

    def __init__(self, max_guesses: int = 6, name: str = "minFDE") -> None:
        super(MinFDE, self).__init__(name)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.max_guesses = max_guesses

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        prob: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        keep_invalid_final_step: bool = True,
    ) -> None:
        pred, target, prob, valid_mask, _ = valid_filter(
            pred, target, prob, valid_mask, keep_invalid_final_step
        )
        pred_topk, _ = topk(self.max_guesses, pred, prob)
        inds_last = (
            valid_mask
            * torch.arange(1, valid_mask.size(-1) + 1, device=pred.device)
        ).argmax(dim=-1)
        self.sum += (
            torch.norm(
                pred_topk[torch.arange(pred.size(0)), :, inds_last]
                - target[torch.arange(pred.size(0)), inds_last].unsqueeze(-2),
                p=2,
                dim=-1,
            )
            .min(dim=-1)[0]
            .sum()
        )
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count


@OBJECT_REGISTRY.register
class MissRate(EvalMetric):
    """
    Miss Rate (MR) evaluation metric for trajectory predictions.

    The MR calculates the rate of missed predictions based on a
    specified criterion and threshold.

    Args:
        max_guesses: Maximum number of guesses to consider for top-k
            predictions.
        name: Name of the metric.
    """

    def __init__(self, max_guesses: int = 6, name: str = "MR") -> None:
        super(MissRate, self).__init__(name)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.max_guesses = max_guesses

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        prob: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        keep_invalid_final_step: bool = True,
        miss_criterion: str = "FDE",
        miss_threshold: float = 2.0,
    ) -> None:
        pred, target, prob, valid_mask, _ = valid_filter(
            pred, target, prob, valid_mask, keep_invalid_final_step
        )
        pred_topk, _ = topk(self.max_guesses, pred, prob)
        if miss_criterion == "FDE":
            inds_last = (
                valid_mask
                * torch.arange(1, valid_mask.size(-1) + 1, device=pred.device)
            ).argmax(dim=-1)
            self.sum += (
                torch.norm(
                    pred_topk[torch.arange(pred.size(0)), :, inds_last]
                    - target[torch.arange(pred.size(0)), inds_last].unsqueeze(
                        -2
                    ),
                    p=2,
                    dim=-1,
                ).min(dim=-1)[0]
                > miss_threshold
            ).sum()
        elif miss_criterion == "MAXDE":
            self.sum += (
                (
                    (
                        torch.norm(
                            pred_topk - target.unsqueeze(1), p=2, dim=-1
                        )
                        * valid_mask.unsqueeze(1)
                    ).max(dim=-1)[0]
                ).min(dim=-1)[0]
                > miss_threshold
            ).sum()
        else:
            raise ValueError(
                "{} is not a valid criterion".format(miss_criterion)
            )
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count


@OBJECT_REGISTRY.register
class HitRate(EvalMetric):
    """
    Hit rate evaluation metric for trajectory predictions.

    Args:
        max_guesses: Maximum number of guesses to consider for top-k
            predictions.
        name: Name of the metric.
    """

    def __init__(
        self,
        max_guesses: int = 6,
        name: str = "HitRate",
    ) -> None:
        super(HitRate, self).__init__(name)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.max_guesses = max_guesses

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        prob: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        keep_invalid_final_step: bool = True,
        miss_criterion: str = "FDE",
        miss_threshold: float = 2.0,
    ) -> None:
        pred, target, prob, valid_mask, _ = valid_filter(
            pred, target, prob, valid_mask, keep_invalid_final_step
        )
        pred_topk, _ = topk(self.max_guesses, pred, prob)
        if miss_criterion == "FDE":
            inds_last = (
                valid_mask
                * torch.arange(1, valid_mask.size(-1) + 1, device=pred.device)
            ).argmax(dim=-1)
            self.sum += (
                torch.norm(
                    pred_topk[torch.arange(pred.size(0)), :, inds_last]
                    - target[torch.arange(pred.size(0)), inds_last].unsqueeze(
                        -2
                    ),
                    p=2,
                    dim=-1,
                ).min(dim=-1)[0]
                <= miss_threshold
            ).sum()
        elif miss_criterion == "MAXDE":
            self.sum += (
                (
                    (
                        torch.norm(
                            pred_topk - target.unsqueeze(1), p=2, dim=-1
                        )
                        * valid_mask.unsqueeze(1)
                    ).max(dim=-1)[0]
                ).min(dim=-1)[0]
                <= miss_threshold
            ).sum()
        else:
            raise ValueError(
                "{} is not a valid criterion".format(miss_criterion)
            )
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
