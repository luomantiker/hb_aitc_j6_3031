import torch

__all__ = ["decode_heatmap"]


def decode_heatmap(
    heatmap: torch.Tensor,
    scale: int,
    mode: str = "diff_sign",
    k_size: int = 5,
):
    """Decode heatmap prediction to landmark coordinates.

    Args:
        heatmap: import heatmap tensor to be decoder, shape: [B, C, H, W].
        scale: Same as feat stride, the Scale of heatmap coordinates
                relative to the original image.
        mode: The decoder method, currently support "diff_sign" and "averaged"
            In the 'averaged' mode, the coordinates and heatmap values of the
                area surrounding the maximum point on the heatmap, with a size
                of k_size x k_size, are weighted to obtain the coordinates
                of the key point.
        k_size: kernel size used for "averaged" decoder.
    """

    B, C, H, W = heatmap.shape
    heatmap1 = heatmap.reshape([B, C, -1])
    conf, max_co = torch.max(heatmap1, dim=2, keepdim=True)

    cy = (max_co / W).long()  # (B, C, 1)
    cx = max_co.long() % W  # (B, C, 1)

    def mat_idx(M, x, y):
        invalid = (x < 0) | (x > (W - 1)) | (y < 0) | (y > H - 1)
        x, y = x.clamp(0, W - 1), y.clamp(0, H - 1)

        res = M.gather(2, y * W + x)
        zero_tensor = torch.zeros(res.shape).to(res.dtype).to(res.device)
        return torch.where(invalid, zero_tensor, res)

    def gen_xy_coord(H, W):
        x_coord = torch.arange(H).unsqueeze(0).repeat(W, 1)
        y_coord = torch.arange(W).unsqueeze(1).repeat(1, H)
        return x_coord, y_coord

    conf = conf.clamp(0, 1)
    if mode == "diff_sign":
        diff_x = mat_idx(heatmap1, cx + 1, cy) - mat_idx(heatmap1, cx - 1, cy)
        diff_y = mat_idx(heatmap1, cx, cy + 1) - mat_idx(heatmap1, cx, cy - 1)
        cx = cx + 0.25 * torch.sign(diff_x)
        cy = cy + 0.25 * torch.sign(diff_y)
        pred_ldmk = torch.cat([cx, cy, conf], dim=2) * scale

    elif mode == "averaged":
        ker_xidx, ker_yidx = gen_xy_coord(k_size, k_size)
        ker_xidx = (ker_xidx - (k_size // 2)).reshape([1, 1, -1]).to(cx.device)
        ker_yidx = (ker_yidx - (k_size // 2)).reshape([1, 1, -1]).to(cx.device)
        # (1, 1, K)

        ker_cx = cx + ker_xidx
        ker_cy = cy + ker_yidx  # (B, C, K)

        ker_val = mat_idx(heatmap1, ker_cx, ker_cy)

        pred_x = (ker_val * ker_cx).sum(dim=2) / (ker_val.sum(dim=2) + 1e-9)
        pred_y = (ker_val * ker_cy).sum(dim=2) / (ker_val.sum(dim=2) + 1e-9)

        pred_x = pred_x.unsqueeze(2)
        pred_y = pred_y.unsqueeze(2)

        pred_ldmk = torch.cat([pred_x, pred_y, conf], dim=2) * scale

    return pred_ldmk
