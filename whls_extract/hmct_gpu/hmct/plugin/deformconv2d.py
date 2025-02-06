from onnxruntime_extensions import PyCustomOpDef, onnx_op
import torch
from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args
import torchvision  # noqa: F401

__all__ = []


@onnx_op(
    op_type="ai.onnx.contrib::DeformConv",
    inputs=[
        PyCustomOpDef.dt_float,
        PyCustomOpDef.dt_float,
        PyCustomOpDef.dt_float,
        PyCustomOpDef.dt_float,
        PyCustomOpDef.dt_float,
        PyCustomOpDef.dt_int64,
        PyCustomOpDef.dt_int64,
        PyCustomOpDef.dt_int64,
        PyCustomOpDef.dt_int64,
        PyCustomOpDef.dt_int64,
        PyCustomOpDef.dt_int64,
        PyCustomOpDef.dt_int64,
        PyCustomOpDef.dt_int64,
        PyCustomOpDef.dt_bool,
    ],
    outputs=[PyCustomOpDef.dt_float],
)
def deform_conv2d_func(
    input,  # noqa: A002
    weight,
    offset,
    mask,
    bias,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dil_h,
    dil_w,
    n_weight_grps,
    n_offset_grps,
    use_mask,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = torch.from_numpy(input).to(device)  # noqa: A001
    weight = torch.from_numpy(weight).to(device)
    offset = torch.from_numpy(offset).to(device)
    bias = torch.from_numpy(bias).to(device)
    mask = torch.from_numpy(mask).to(device)
    res = torch.ops.torchvision.deform_conv2d(
        input,
        weight,
        offset,
        mask,
        bias,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dil_h,
        dil_w,
        n_weight_grps,
        n_offset_grps,
        use_mask,
    )
    return res.cpu().numpy()


@parse_args("v", "v", "v", "v", "v", "i", "i", "i", "i", "i", "i", "i", "i", "b")
def deform_conv2d_symbolic(
    g,
    input,  # noqa: A002
    weight,
    offset,
    mask,
    bias,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dil_h,
    dil_w,
    n_weight_grps,
    n_offset_grps,
    use_mask,
):
    # calculate the output shape
    output_shape = offset.type().sizes()
    output_shape[1] = weight.type().sizes()[0]

    stride_h = g.op("Constant", value_t=torch.tensor(stride_h))
    stride_w = g.op("Constant", value_t=torch.tensor(stride_w))
    pad_h = g.op("Constant", value_t=torch.tensor(pad_h))
    pad_w = g.op("Constant", value_t=torch.tensor(pad_w))
    dil_h = g.op("Constant", value_t=torch.tensor(dil_h))
    dil_w = g.op("Constant", value_t=torch.tensor(dil_w))
    n_weight_grps = g.op("Constant", value_t=torch.tensor(n_weight_grps))
    n_offset_grps = g.op("Constant", value_t=torch.tensor(n_offset_grps))
    use_mask = g.op("Constant", value_t=torch.tensor(use_mask))

    return g.op(
        "ai.onnx.contrib::DeformConv",
        input,
        weight,
        offset,
        mask,
        bias,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dil_h,
        dil_w,
        n_weight_grps,
        n_offset_grps,
        use_mask,
    ).setType(input.type().with_dtype(torch.float32).with_sizes(output_shape))


register_custom_op_symbolic("torchvision::deform_conv2d", deform_conv2d_symbolic, 10)
