import torch

from hat.registry import OBJECT_REGISTRY

__all__ = ["GroundBox3dCoder"]


@OBJECT_REGISTRY.register
class GroundBox3dCoder:
    """Box3d Coder for Lidar.

    Args:
        linear_dim: Whether to smooth dimension. Defaults to False.
        vec_encode: Whether encode angle to vector. Defaults to False.
        n_dim: dims of bbox3d. Defaults to 7.
        norm_velo: Whether to normalize. Defaults to False.
    """

    def __init__(
        self,
        linear_dim: bool = False,
        vec_encode: bool = False,
        n_dim: int = 7,
        norm_velo: bool = False,
    ):

        super().__init__()
        self.linear_dim = linear_dim
        self.vec_encode = vec_encode
        self.norm_velo = norm_velo
        self.n_dim = n_dim

    @property
    def code_size(self):
        return self.n_dim + 1 if self.vec_encode else self.n_dim

    def encode(
        self,
        boxes: torch.Tensor,
        anchors: torch.Tensor,
    ):
        """Box encode for Lidar boxes.

        Args:
            boxes: normal boxes, shape [N, 7]: x, y, z, l, w, h, r
            anchors: anchors, shape [N, 7]: x, y, z, l, w, h, r
        """
        box_ndim = anchors.shape[-1]

        if box_ndim == 7:
            xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
            xg, yg, zg, wg, lg, hg, rg = torch.split(boxes, 1, dim=-1)
        else:
            xa, ya, za, wa, la, ha, vxa, vya, ra = torch.split(
                anchors, 1, dim=-1
            )
            xg, yg, zg, wg, lg, hg, vxg, vyg, rg = torch.split(
                boxes, 1, dim=-1
            )

        diagonal = torch.sqrt(la ** 2 + wa ** 2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / ha

        if self.linear_dim:
            lt = lg / la - 1
            wt = wg / wa - 1
            ht = hg / ha - 1
        else:
            lt = torch.log(lg / la)
            wt = torch.log(wg / wa)
            ht = torch.log(hg / ha)

        ret = [xt, yt, zt, wt, lt, ht]

        if box_ndim > 7:
            if self.norm_velo:
                vxt = (vxg - vxa) / diagonal
                vyt = (vyg - vya) / diagonal
            else:
                vxt = vxg - vxa
                vyt = vyg - vya
            ret.extend([vxt, vyt])

        if self.vec_encode:
            rgx = torch.cos(rg)
            rgy = torch.sin(rg)
            rax = torch.cos(ra)
            ray = torch.sin(ra)
            rtx = rgx - rax
            rty = rgy - ray
            ret.extend([rtx, rty])
        else:
            rt = rg - ra
            ret.append(rt)

        return torch.cat(ret, dim=-1)

    def decode(
        self,
        box_encodings,
        anchors,
    ):
        """Box decode for lidar bbox.

        Args:
            boxes: normal boxes, shape [N, 7]: (x, y, z, w, l, h, r)
            anchors: anchors, shape [N, 7]: (x, y, z, w, l, h, r)
        """
        box_ndim = anchors.shape[-1]

        if box_ndim == 9:
            xa, ya, za, wa, la, ha, vxa, vya, ra = torch.split(
                anchors, 1, dim=-1
            )
            if self.vec_encode:
                xt, yt, zt, wt, lt, ht, vxt, vyt, rtx, rty = torch.split(
                    box_encodings, 1, dim=-1
                )
            else:
                xt, yt, zt, wt, lt, ht, vxt, vyt, rt = torch.split(
                    box_encodings, 1, dim=-1
                )
        elif box_ndim == 7:
            xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
            if self.vec_encode:
                xt, yt, zt, wt, lt, ht, rtx, rty = torch.split(
                    box_encodings, 1, dim=-1
                )
            else:
                xt, yt, zt, wt, lt, ht, rt = torch.split(
                    box_encodings, 1, dim=-1
                )

        diagonal = torch.sqrt(la ** 2 + wa ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za

        ret = [xg, yg, zg]

        if self.linear_dim:
            lg = (lt + 1) * la
            wg = (wt + 1) * wa
            hg = (ht + 1) * ha
        else:

            lg = torch.exp(lt) * la
            wg = torch.exp(wt) * wa
            hg = torch.exp(ht) * ha
        ret.extend([wg, lg, hg])

        if self.vec_encode:
            rax = torch.cos(ra)
            ray = torch.sin(ra)
            rgx = rtx + rax
            rgy = rty + ray
            rg = torch.atan2(rgy, rgx)
        else:
            rg = rt + ra

        if box_ndim > 7:
            if self.norm_velo:
                vxg = vxt * diagonal + vxa
                vyg = vyt * diagonal + vya
            else:
                vxg = vxt + vxa
                vyg = vyt + vya
            ret.extend([vxg, vyg])

        ret.append(rg)

        return torch.cat(ret, dim=-1)
