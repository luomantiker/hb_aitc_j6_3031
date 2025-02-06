# Copyright (c) Horizon Robotics. All rights reserved.

import numpy as np
from shapely.geometry.polygon import Polygon

from hat.core.box_utils import bbox_overlaps


class QuickUnionFind:
    def __init__(self, size):
        self.size = size
        self.count = size
        self._parent = list(range(self.size))

    def find(self, p):
        assert 0 <= p < self.size
        if p != self._parent[p]:
            self._parent[p] = self.find(self._parent[p])
        return self._parent[p]

    def connected(self, p, q):
        return self.find(p) == self.find(q)

    def union(self, p, q):
        root_p = self.find(p)
        root_q = self.find(q)
        if root_p != root_q:
            self._parent[root_p] = root_q
            self.count -= 1


def bbox_to_polygon(bbox):
    x1, y1, x2, y2 = bbox[:4]
    x2 += 1.0
    y2 += 1.0
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


def multi_bboxes_to_polygon(bboxes):
    p = Polygon()
    for bbox in bboxes:
        p = p.union(bbox_to_polygon(bbox))
    return p


def get_crowd_groups(bboxes, iou_thresh=1e-6):
    uf = QuickUnionFind(len(bboxes))
    iou_matrix = bbox_overlaps(bboxes, bboxes)
    lhs_idxs, rhs_idxs = np.where(iou_matrix >= iou_thresh)
    unique_idxs = lhs_idxs <= rhs_idxs
    lhs_idxs = lhs_idxs[unique_idxs]
    rhs_idxs = rhs_idxs[unique_idxs]
    for lhs_idx, rhs_idx in zip(lhs_idxs, rhs_idxs):
        if not uf.connected(lhs_idx, rhs_idx):
            uf.union(lhs_idx, rhs_idx)

    groups = {}
    for i in range(uf.size):
        # root_idx = uf.find(i)
        groups.setdefault(uf.find(i), []).append(i)
    return list(map(lambda k: groups[k], groups))
