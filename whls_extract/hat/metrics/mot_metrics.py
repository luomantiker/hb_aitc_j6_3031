# Copyright (c) Horizon Robotics. All rights reserved.


import glob
import logging
import os
import shutil
from collections import OrderedDict
from os import path as osp
from pathlib import Path
from typing import Dict

try:
    import motmetrics as mm
except ImportError:
    mm = None

from hat.registry import OBJECT_REGISTRY
from hat.utils.distributed import get_dist_info
from hat.utils.package_helper import require_packages
from .metric import EvalMetric

__all__ = ["MotMetric"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class MotMetric(EvalMetric):
    """Evaluation in MOT.

    Args:
        gt_dir: validation data gt dir.
        name: name of this metric instance for display.
        save_prefix: path to save result.
        cleanup: whether to clean up the saved results when the process ends.

    """

    @require_packages("motmetrics")
    def __init__(
        self,
        gt_dir: str,
        name: str = "MOTA",
        save_prefix: str = "./WORKSPACE/motresults",
        cleanup: bool = False,
    ):
        super().__init__(name)
        self.cleanup = cleanup

        rank, world_size = get_dist_info()
        self.save_prefix = save_prefix + str(rank) + "_" + str(world_size)

        try:
            os.makedirs(osp.expanduser(self.save_prefix))
        except FileExistsError:
            shutil.rmtree(osp.expanduser(self.save_prefix))
            os.makedirs(osp.expanduser(self.save_prefix))

        self.seq_name_list = []
        self.metric_idx = 0

        self.gtfiles = glob.glob(os.path.join(gt_dir, "*/gt/gt.txt"))
        self.gt = OrderedDict(
            [
                (
                    Path(f).parts[-3],
                    mm.io.loadtxt(f, fmt="mot15-2D", min_confidence=1),
                )
                for f in self.gtfiles
            ]
        )
        self.mh = mm.metrics.create()
        self.formatters = self.mh.formatters
        self.formatters["mota"] = "{:.4f}".format

        self.name = name
        self.value = 0.0

    def _init_states(self):
        self._results = []

    def __del__(self):
        if self.cleanup:
            try:
                shutil.rmtree(osp.expanduser(self.save_prefix))
            except IOError as err:
                logger.error(str(err))

    def reset(self):
        self._results = []
        self.seq_name_list = []

    def get(self):
        """Get evaluation metrics."""
        filenames = []
        dir_each = osp.join(self.save_prefix, f"_{self.metric_idx}")
        save_format = "{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n"
        for each_result in self._results:
            seq_name = each_result["seq_name"]
            seq_result_path = osp.join(dir_each, seq_name)
            os.makedirs(seq_result_path, exist_ok=True)
            filename = osp.join(seq_result_path, "gt.txt")
            if not os.path.exists(filename):
                f = open(filename, "w")
                filenames.append(filename)
            else:
                f = open(filename, "a")
            for xyxy, track_id in zip(
                each_result["pred_boxes"], each_result["obj_idxes"]
            ):
                if track_id < 0 or track_id is None:
                    continue
                x1, y1, x2, y2 = xyxy
                w, h = x2 - x1, y2 - y1
                line = save_format.format(
                    frame=int(each_result["seq_frame_id"]),
                    id=int(track_id + 1),
                    x1=x1,
                    y1=y1,
                    w=w,
                    h=h,
                )
                f.write(line)
            f.close()

        self.metric_idx += 1

        ts = OrderedDict(
            [
                (Path(f).parts[-2], mm.io.loadtxt(f, fmt="mot15-2D"))
                for f in filenames
            ]
        )
        accs, names = self._compare_dataframes(self.gt, ts)
        metrics = list(mm.metrics.motchallenge_metrics)
        summary = self.mh.compute_many(
            accs, names=names, metrics=metrics, generate_overall=True
        )
        result = mm.io.render_summary(
            summary,
            formatters=self.formatters,
            namemap=mm.io.motchallenge_metric_names,
        )
        value = result.split("\n")[-1].split()[14]
        self.value = float(value)
        return self.name, self.value

    def _compare_dataframes(self, gts, ts):
        """Build accumulator for each sequence."""
        accs = []
        names = []
        for k, tsacc in ts.items():
            if k in gts:
                # logger.info('Comparing %s...', k)
                accs.append(
                    mm.utils.compare_to_groundtruth(
                        gts[k], tsacc, "iou", distth=0.5
                    )
                )
                names.append(k)
            else:
                logger.warning("No ground truth for %s, skipping.", k)

        return accs, names

    def update(self, outputs: Dict):
        """Update internal buffer with latest predictions.

        Note that the statistics are not available until
        you call self.get() to return the metrics.

        Args:
            output: A dict of model output which includes det results and
                image infos.

        """

        for _, output in outputs.items():
            seq_name = output["seq_name"]

            if seq_name not in self.seq_name_list:
                self.seq_name_list.append(seq_name)

            self._results.append(
                {
                    "pred_boxes": output["pred_boxes"].cpu().numpy(),
                    "obj_idxes": output["obj_idxes"].cpu().numpy(),
                    "seq_frame_id": output["seq_frame_id"],
                    "seq_name": seq_name,
                }
            )
