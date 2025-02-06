# Copyright (c) Horizon Robotics. All rights reserved.

import logging
import math
from typing import Dict, List, Optional

import numpy as np

from hat.registry import OBJECT_REGISTRY
from hat.utils.distributed import (
    all_gather_object,
    get_dist_info,
    rank_zero_only,
)
from .metric import EvalMetric

logger = logging.getLogger(__name__)

__all__ = ["ArgoverseMetric"]

LOW_PROB_THRESHOLD_FOR_METRICS = 0.05


@OBJECT_REGISTRY.register
class ArgoverseMetric(EvalMetric):
    """Evaluation Argoverse Detection.

    Args:
        name: Name of this metric instance for display.
        max_guesses: Number of guesses allowed.
        horizon: Prediction horizon.
        miss_threshold: Distance threshold for
                        the last predicted coordinate.
    """

    def __init__(
        self,
        name: str = "ArgoverseMetric",
        max_guesses: int = 6,
        horizon: int = 30,
        miss_threshold: float = 2.0,
    ):
        super(ArgoverseMetric, self).__init__(name)
        self.ret = ["minFDE", 0.0]
        self.max_guesses = max_guesses
        self.horizon = horizon
        self.miss_threshold = miss_threshold
        self.metrics = {}

    def compute(self):
        pass

    def get_ade(
        self, forecasted_trajectory: np.ndarray, gt_trajectory: np.ndarray
    ) -> float:
        """Compute Average Displacement Error.

        Args:
            forecasted_trajectory: Predicted trajectory with shape.
                                   (pred_len x 2)
            gt_trajectory: Ground truth trajectory with shape.
                                   (pred_len x 2)

        Returns:
            ade: Average Displacement Error

        """
        pred_len = forecasted_trajectory.shape[0]
        ade = float(
            sum(
                math.sqrt(
                    (forecasted_trajectory[i, 0] - gt_trajectory[i, 0]) ** 2
                    + (forecasted_trajectory[i, 1] - gt_trajectory[i, 1]) ** 2
                )
                for i in range(pred_len)
            )
            / pred_len
        )
        return ade

    def get_fde(
        self, forecasted_trajectory: np.ndarray, gt_trajectory: np.ndarray
    ) -> float:
        """Compute Final Displacement Error.

        Args:
            forecasted_trajectory: Predicted trajectory with shape.
                                   (pred_len x 2)
            gt_trajectory: Ground truth trajectory with shape.
                                   (pred_len x 2)

        Returns:
            fde: Final Displacement Error

        """
        fde = math.sqrt(
            (forecasted_trajectory[-1, 0] - gt_trajectory[-1, 0]) ** 2
            + (forecasted_trajectory[-1, 1] - gt_trajectory[-1, 1]) ** 2
        )
        return fde

    def get_displacement_errors_and_miss_rate(
        self,
        forecasted_trajectories: Dict[int, List[np.ndarray]],
        gt_trajectories: Dict[int, np.ndarray],
        forecasted_probabilities: Optional[Dict[int, List[float]]] = None,
    ) -> Dict[str, float]:
        metric_results: Dict[str, float] = {}
        min_ade, prob_min_ade, brier_min_ade = [], [], []
        min_fde, prob_min_fde, brier_min_fde = [], [], []
        n_misses, prob_n_misses = [], []

        for k, v in gt_trajectories.items():
            curr_min_ade = float("inf")
            curr_min_fde = float("inf")
            min_idx = 0
            max_num_traj = min(
                self.max_guesses, len(forecasted_trajectories[k])
            )

            if forecasted_probabilities is not None:
                sorted_idx = np.argsort(
                    [-x for x in forecasted_probabilities[k]], kind="stable"
                )
                pruned_probabilities = [
                    forecasted_probabilities[k][t]
                    for t in sorted_idx[:max_num_traj]
                ]
                # Normalize
                prob_sum = sum(pruned_probabilities)
                pruned_probabilities = [
                    p / prob_sum for p in pruned_probabilities
                ]
            else:
                sorted_idx = np.arange(len(forecasted_trajectories[k]))
            pruned_trajectories = [
                forecasted_trajectories[k][t]
                for t in sorted_idx[:max_num_traj]
            ]

            for j in range(len(pruned_trajectories)):
                fde = self.get_fde(
                    pruned_trajectories[j][: self.horizon], v[: self.horizon]
                )
                if fde < curr_min_fde:
                    min_idx = j
                    curr_min_fde = fde
            curr_min_ade = self.get_ade(
                pruned_trajectories[min_idx][: self.horizon], v[: self.horizon]
            )
            min_ade.append(curr_min_ade)
            min_fde.append(curr_min_fde)
            n_misses.append(curr_min_fde > self.miss_threshold)

            if forecasted_probabilities is not None:
                prob_n_misses.append(
                    1.0
                    if curr_min_fde > self.miss_threshold
                    else (1.0 - pruned_probabilities[min_idx])
                )
                prob_min_ade.append(
                    min(
                        -np.log(pruned_probabilities[min_idx]),
                        -np.log(LOW_PROB_THRESHOLD_FOR_METRICS),
                    )
                    + curr_min_ade
                )
                brier_min_ade.append(
                    (1 - pruned_probabilities[min_idx]) ** 2 + curr_min_ade
                )
                prob_min_fde.append(
                    min(
                        -np.log(pruned_probabilities[min_idx]),
                        -np.log(LOW_PROB_THRESHOLD_FOR_METRICS),
                    )
                    + curr_min_fde
                )
                brier_min_fde.append(
                    (1 - pruned_probabilities[min_idx]) ** 2 + curr_min_fde
                )

        metric_results["minADE"] = sum(min_ade)
        metric_results["minFDE"] = sum(min_fde)
        metric_results["MR"] = sum(n_misses)
        metric_results["minADE_len"] = len(min_ade)
        metric_results["minFDE_len"] = len(min_fde)
        metric_results["MR_len"] = len(n_misses)

        if forecasted_probabilities is not None:
            metric_results["p-minADE"] = sum(prob_min_ade)
            metric_results["p-minFDE"] = sum(prob_min_fde)
            metric_results["p-MR"] = sum(prob_n_misses)
            metric_results["brier-minADE"] = sum(brier_min_ade)
            metric_results["brier-minFDE"] = sum(brier_min_fde)
            metric_results["p-minADE_len"] = len(prob_min_ade)
            metric_results["p-minFDE_len"] = len(prob_min_fde)
            metric_results["p-MR_len"] = len(prob_n_misses)
            metric_results["brier-minADE_len"] = len(brier_min_ade)
            metric_results["brier-minFDE_len"] = len(brier_min_fde)
        return metric_results

    def update(
        self,
        meta,
        preds,
    ):
        pred_trajs, scores = preds
        file_names = meta["file_name"]
        gts = meta["traj_labels"].cpu().numpy()

        traj_pred = {}
        traj_gts = {}
        for gt, pred_traj, fs in zip(gts, pred_trajs, file_names):
            traj_pred[fs] = pred_traj
            traj_gts[fs] = gt
        metric = self.get_displacement_errors_and_miss_rate(
            traj_pred, traj_gts
        )
        for k, v in metric.items():
            if k not in self.metrics:
                self.metrics[k] = v
            else:
                self.metrics[k] += v

    def _gather(self):
        global_rank, global_world_size = get_dist_info()
        global_output = [None for _ in range(global_world_size)]
        all_gather_object(global_output, self.metrics)
        output = global_output[0]
        for out in global_output[1:]:
            for k, v in out.items():
                output[k] += v
        result = {}
        for k, v in output.items():
            if "_len" not in k:
                k_len = f"{k}_len"
                leng = output[k_len]
                result[k] = v / leng
        return result

    def reset(self):
        self.metrics = {}
        self.ret = ["minFDE", 0.0]

    def get(self):
        metrics = self._gather()
        self._get(metrics)
        return self.ret[0], self.ret[1]

    @rank_zero_only
    def _get(self, metric):
        logger.info(f"The whole metric: {metric}")
        if "minFDE" in metric:
            self.ret[1] = metric["minFDE"]
