import torch


def proj_func(xyz, K):
    """Proj func of hand landmarks.

    Args:
        xyz: N x num_points x 3
        K: N x 3 x 3
    """
    uv = torch.bmm(K, xyz.permute(0, 2, 1))
    uv = uv.permute(0, 2, 1)
    out_uv = torch.zeros_like(uv[:, :, :2]).to(device=uv.device)
    out_uv = torch.addcdiv(
        out_uv, uv[:, :, :2], uv[:, :, 2].unsqueeze(-1).repeat(1, 1, 2)
    )
    return out_uv


def posenc(input_tensor, num_encodings=4):
    """
    Positional encoding function for NeRF models.

    Args:
        input_tensor: Input tensor of shape (
            batch_size, seq_len, emb_dim
        ).
        num_encodings: Number of encoding functions to use.

    Returns:
        output_tensor: Output tensor of shape (
                batch_size, seq_len,
                emb_dim + 2 * num_encodings * emb_dim,
        ).
    """
    output_tensor = [input_tensor] + [
        encoding_func(2.0 ** i * input_tensor)
        for encoding_func in (torch.sin, torch.cos)
        for i in range(num_encodings)
    ]
    return torch.cat(output_tensor, -1)
