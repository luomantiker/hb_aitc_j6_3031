from copy import deepcopy

import torch

from .box_utils import bbox_overlaps


def take_results_by_key(results, keys):
    if not isinstance(keys, (tuple, list)):
        keys = [keys]
    ret = []
    for key in keys:
        ret.append(results[key])
    return ret


class PersonROI3DFilterAdapter(object):
    def __init__(
        self,
        take_keys,
        iou_thresh,
        cyc_thresh,
    ):
        self.take_keys = take_keys
        self.iou_thresh = iou_thresh
        self.cyc_thresh = cyc_thresh

    def _match_to_cyclist(self, person, cyclist_all):
        for i in range(cyclist_all.boxes.shape[0]):
            cyc_score = cyclist_all[i].score
            cyc_bbox = cyclist_all[i].box
            if cyc_score > self.cyc_thresh:
                if (
                    bbox_overlaps(
                        person.box.unsqueeze(0),
                        cyc_bbox.unsqueeze(0),
                        mode="iof",
                    )[0, 0]
                    > self.iou_thresh
                ):
                    return True
        return False

    def __call__(self, perception):
        (perception,) = take_results_by_key(perception, self.take_keys)
        batch_size = len(perception["person"])
        ped_perception = deepcopy(perception["person"])
        cyc_perception = perception["cyclist"]
        for batch_idx in range(batch_size):
            ped_det = ped_perception[batch_idx].person_detection
            cyc_det = cyc_perception[batch_idx].cyclist_detection
            num_ped_roi = ped_det.boxes.shape[0]
            keep = []
            for i in range(num_ped_roi):
                # filter person roi if person match cyclist
                keep.append(not self._match_to_cyclist(ped_det[i], cyc_det))
            ped_perception[batch_idx] = ped_perception[batch_idx][
                torch.BoolTensor(keep)
            ]
        perception["person"] = ped_perception
        return perception


class BBoxScoreFilterAdapter(object):
    def __init__(
        self,
        take_keys,
        box_score_threshold,
        filtered_obj=None,
    ):
        self.take_keys = take_keys
        self.box_score_threshold = box_score_threshold
        self.filtered_obj = filtered_obj

    def __call__(self, perception):
        (obj_key, det_key, perception) = take_results_by_key(
            perception, self.take_keys
        )
        if isinstance(obj_key, (tuple, list)):
            if self.filtered_obj is not None:
                _tmp = []
                for k in obj_key:
                    if k in self.filtered_obj:
                        _tmp.append(k)
                obj_key = _tmp
        else:
            obj_key = [obj_key]

        keep_perception = deepcopy(perception)
        for obj in obj_key:
            keep_perception[obj] = []
            for batch in perception[obj]:
                batch = batch.filter_by_lambda(
                    lambda x: getattr(x, det_key).scores
                    >= self.box_score_threshold
                )
                keep_perception[obj].append(batch)

        return keep_perception
