patch_hz_ops_metadata = [
    {
        "name": "ops.horizon.convert_conv_params",
        "inputs": [
            {"name": "input_scale", "type": "Tensor"},
            {"name": "weight", "type": "Tensor"},
            {"name": "weight_scale", "type": "Tensor"},
            {"name": "weight_dtype", "type": "string"},
            {"name": "bias", "type": "Tensor"},
            {"name": "bias_scale", "type": "Tensor"},
            {"name": "bias_dtype", "type": "string"},
            {"name": "out_scale", "type": "Tensor"},
            {"name": "out_dtype", "type": "string"},
            {"name": "edata_scale", "type": "Tensor"},
            {"name": "is_conv_transpose2d", "type": "boolean"},
            {"name": "groups", "type": "int64"},
            {"name": "march_str", "type": "string"},
        ],
        "outputs": [
            {"name": "bpu_weight", "type": "Tensor"},
            {"name": "bpu_bias", "type": "Tensor"},
            {"name": "bpu_bias_lshift", "type": "Tensor"},
            {"name": "bpu_escale", "type": "Tensor"},
            {"name": "bpu_escale_lshift", "type": "Tensor"},
            {"name": "bpu_oscale", "type": "Tensor"},
            {"name": "bpu_accu_rshift", "type": "Tensor"},
            {"name": "bpu_output_rshift", "type": "Tensor"},
            {"name": "dequant_output_scale", "type": "Tensor"},
        ],
    },
    {
        "name": "ops.horizon.bpu_scale_quantization",
        "inputs": [
            {"name": "data", "type": "Tensor"},
            {"name": "scale", "type": "Tensor"},
            {"name": "zero_point", "type": "Tensor"},
            {"name": "vector_dim", "type": "int64"},
            {"name": "quant_min", "type": "int64"},
            {"name": "quant_max", "type": "int64"},
            {"name": "qtype", "type": "string"},
            {"name": "round_mode", "type": "string"},
            {"name": "march_str", "type": "string"},
        ],
        "outputs": [{"name": "output", "type": "Tensor"}],
    },
    {
        "name": "ops.horizon.bpu_scale_dequantization",
        "inputs": [
            {"name": "data", "type": "Tensor"},
            {"name": "scale", "type": "Tensor"},
            {"name": "ch_axis", "type": "int64"},
        ],
        "outputs": [{"name": "output", "type": "Tensor"}],
    },
    {
        "name": "ops.horizon.bpu_scale_quanti_convolution",
        "inputs": [
            {"name": "data_", "type": "Tensor"},
            {"name": "weight_", "type": "Tensor"},
            {"name": "bias_", "type": "Tensor"},
            {"name": "sumin_", "type": "Tensor"},
            {"name": "output_scale_", "type": "Tensor"},
            {"name": "accu_right_shift_", "type": "Tensor"},
            {"name": "bias_left_shift_", "type": "Tensor"},
            {"name": "output_right_shift_", "type": "Tensor"},
            {"name": "sumin_scale_", "type": "Tensor"},
            {"name": "sumin_left_shift_", "type": "Tensor"},
            {"name": "use_bias", "type": "boolean"},
            {"name": "filters", "type": "int64"},
            {"name": "kernel_size", "type": "int64[]"},
            {"name": "strides", "type": "int64[]"},
            {"name": "pads", "type": "int64[]"},
            {"name": "dilation_rate", "type": "int64[]"},
            {"name": "activation", "type": "string"},
            {"name": "group", "type": "int64"},
            {"name": "elementwise_input", "type": "boolean"},
            {"name": "disable_output_quantization", "type": "boolean"},
            {"name": "out_quanti_type", "type": "string"},
            {"name": "march_str", "type": "string"},
        ],
        "outputs": [{"name": "output", "type": "Tensor"}],
    },
    {
        "name": "ops.horizon.bpu_post_process_channel_argmax",
        "inputs": [
            {"name": "input", "type": "Tensor"},
            {"name": "group", "type": "int64"},
            {"name": "march_str", "type": "string"},
        ],
        "outputs": [{"name": "output", "type": "Tensor"}],
    },
    {
        "name": "ops.horizon.bpu_quanti_grid_sample",
        "inputs": [
            {"name": "in_featuremap", "type": "Tensor"},
            {"name": "in_grid", "type": "Tensor"},
            {"name": "mode", "type": "string"},
            {"name": "padding_mode", "type": "string"},
            {"name": "align_corners", "type": "boolean"},
            {"name": "coord_shift", "type": "int64"},
            {"name": "march_str", "type": "string"},
        ],
        "outputs": [{"name": "output", "type": "Tensor"}],
    },
    {
        "name": "ops.horizon.bpu_quanti_line_fit",
        "inputs": [
            {"name": "data", "type": "Tensor"},
            {"name": "otype", "type": "string"},
            {"name": "scale", "type": "int64"},
            {"name": "bias", "type": "int64"},
            {"name": "pshift", "type": "int64"},
        ],
        "outputs": [{"name": "output", "type": "Tensor"}],
    },
    {
        "name": "ops.horizon.bpu_quanti_lut",
        "inputs": [
            {"name": "data", "type": "Tensor"},
            {"name": "table", "type": "Tensor"},
            {"name": "otype", "type": "string"},
            {"name": "scale", "type": "int64"},
            {"name": "bias", "type": "int64"},
            {"name": "pshift", "type": "int64"},
            {"name": "itplt_shift", "type": "int64"},
        ],
        "outputs": [{"name": "output", "type": "Tensor"}],
    },
    {
        "name": "ops.horizon.bpu_quanti_mul",
        "inputs": [
            {"name": "input0", "type": "Tensor"},
            {"name": "input1", "type": "Tensor"},
            {"name": "input_shift0_", "type": "Tensor"},
            {"name": "input_shift1_", "type": "Tensor"},
            {"name": "output_shift_", "type": "Tensor"},
            {"name": "march_str", "type": "string"},
        ],
        "outputs": [{"name": "output", "type": "Tensor"}],
    },
    {
        "name": "ops.horizon.bpu_quanti_proposal",
        "inputs": [
            {"name": "data", "type": "Tensor[]"},
            {"name": "anchor", "type": "Tensor"},
            {"name": "exp_table", "type": "Tensor"},
            {"name": "image_sizes", "type": "Tensor"},
            {"name": "num_anchors", "type": "int64[]"},
            {"name": "num_classes", "type": "int64[]"},
            {"name": "input_shifts", "type": "int64[]"},
            {"name": "exp_shift", "type": "int64[]"},
            {"name": "block_heights", "type": "int64[]"},
            {"name": "block_widths", "type": "int64[]"},
            {"name": "class_output_offsets", "type": "int64"},
            {"name": "random_seed", "type": "int64"},
            {"name": "anchor_start_offsets", "type": "int64[]"},
            {"name": "stride_heights", "type": "int64[]"},
            {"name": "stride_widths", "type": "int64[]"},
            {"name": "use_clippings", "type": "boolean"},
            {"name": "image_size_fixed", "type": "boolean"},
            {"name": "image_height", "type": "int64"},
            {"name": "image_width", "type": "int64"},
            {"name": "im_info_type", "type": "string"},
            {"name": "nms_threshold", "type": "int64"},
            {"name": "output_bbox_num", "type": "int64"},
            {"name": "nms_supress_margin", "type": "int64"},
            {"name": "fake_data_value", "type": "int64"},
            {"name": "march_str", "type": "string"},
        ],
        "outputs": [{"name": "output", "type": "Tensor"}],
    },
    {
        "name": "ops.horizon.quantized_resize",
        "inputs": [
            {"name": "data", "type": "Tensor"},
            {"name": "mode", "type": "string"},
            {"name": "align_corners", "type": "boolean"},
            {"name": "out_height", "type": "int64"},
            {"name": "out_width", "type": "int64"},
            {"name": "ratio_height", "type": "float64"},
            {"name": "ratio_width", "type": "float64"},
            {"name": "march_str", "type": "string"},
        ],
        "outputs": [{"name": "output", "type": "Tensor"}],
    },
    {
        "name": "ops.horizon.quantized_roi_resize",
        "inputs": [
            {"name": "in_featuremap", "type": "Tensor"},
            {"name": "in_rois", "type": "Tensor"},
            {"name": "spatial_scale", "type": "float64"},
            {"name": "out_height", "type": "int64"},
            {"name": "out_width", "type": "int64"},
            {"name": "aligned", "type": "boolean"},
            {"name": "interpolate_mode", "type": "string"},
            {"name": "march_str", "type": "string"},
        ],
        "outputs": [{"name": "output", "type": "Tensor"}],
    },
    {
        "name": "ops.horizon.bpu_scale_quanti_pooling",
        "inputs": [
            {"name": "input_", "type": "Tensor"},
            {"name": "pool_type", "type": "string"},
            {"name": "pool_size", "type": "int64[]"},
            {"name": "pads", "type": "int64"},
            {"name": "strides", "type": "int64"},
            {"name": "ceil_mode", "type": "boolean"},
            {"name": "out_quanti_type", "type": "string"},
            {"name": "march_str", "type": "string"},
        ],
        "outputs": [{"name": "output", "type": "Tensor"}],
    },
    {
        "name": "ops.horizon.bpu_scale_quanti_pooling",
        "inputs": [
            {"name": "input_", "type": "Tensor"},
            {"name": "pool_type", "type": "string"},
            {"name": "pool_size", "type": "int64[]"},
            {"name": "pads", "type": "int64"},
            {"name": "strides", "type": "int64"},
            {"name": "ceil_mode", "type": "boolean"},
            {"name": "out_quanti_type", "type": "string"},
            {"name": "march_str", "type": "string"},
        ],
        "outputs": [{"name": "output", "type": "Tensor"}],
    },
    {
        "name": "ops.horizon.bpu_scale_requantization",
        "inputs": [
            {"name": "data", "type": "Tensor"},
            {"name": "in_scale", "type": "Tensor"},
            {"name": "out_scale", "type": "Tensor"},
            {"name": "input_quanti_type", "type": "string"},
            {"name": "output_quanti_type", "type": "string"},
            {"name": "pre_rshift_with_round", "type": "boolean"},
            {"name": "post_rshift_with_round", "type": "boolean"},
            {"name": "march_str", "type": "string"},
        ],
        "outputs": [{"name": "output", "type": "Tensor"}],
    },
    {
        "name": "ops.horizon.bpu_segment_lut",
        "inputs": [
            {"name": "data", "type": "Tensor"},
            {"name": "table", "type": "Tensor"},
            {"name": "scale", "type": "Tensor"},
            {"name": "beta", "type": "Tensor"},
            {"name": "left_shift", "type": "Tensor"},
            {"name": "right_shift", "type": "Tensor"},
            {"name": "max", "type": "Tensor"},
            {"name": "is_symmetrical", "type": "boolean"},
            {"name": "idx_bits", "type": "int64"},
            {"name": "otype", "type": "string"},
            {"name": "march_str", "type": "string"},
        ],
        "outputs": [{"name": "output", "type": "Tensor"}],
    },
    {
        "name": "ops.horizon.nms_op",
        "inputs": [
            {"name": "boxes", "type": "Tensor"},
            {"name": "scores", "type": "Tensor"},
            {"name": "cls_idxs", "type": "Tensor"},
            {"name": "iou_threshold", "type": "float64"},
            {"name": "pre_top_n", "type": "int64"},
            {"name": "post_top_n", "type": "int64"},
            {"name": "multi_class", "type": "boolean"},
            {"name": "legacy_bbox", "type": "boolean"},
            {"name": "pad_mode", "type": "string"},
        ],
        "outputs": [{"name": "output", "type": "Tensor"}],
    },
    {
        "name": "ops.horizon.quanti_grid_sample",
        "inputs": [
            {"name": "featuremap", "type": "Tensor"},
            {"name": "grid", "type": "Tensor"},
            {"name": "scale", "type": "Tensor"},
            {"name": "mode", "type": "string"},
            {"name": "padding_mode", "type": "string"},
            {"name": "align_corners", "type": "boolean"},
            {"name": "coord_shift", "type": "int64"},
            {"name": "march_str", "type": "string"},
        ],
        "outputs": [{"name": "output", "type": "Tensor"}],
    },
    {
        "name": "ops.horizon.quanti_resize",
        "inputs": [
            {"name": "data", "type": "Tensor"},
            {"name": "scale", "type": "Tensor"},
            {"name": "zero_point", "type": "Tensor"},
            {"name": "mode", "type": "string"},
            {"name": "align_corners", "type": "boolean"},
            {"name": "out_height", "type": "int64"},
            {"name": "out_width", "type": "int64"},
            {"name": "ratio_height", "type": "float64"},
            {"name": "ratio_width", "type": "float64"},
            {"name": "quantized_forward", "type": "boolean"},
            {"name": "march_str", "type": "string"},
        ],
        "outputs": [{"name": "output", "type": "Tensor"}],
    },
]
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402

from packaging import version  # noqa: E402

logger = logging.getLogger(__name__)

try:
    import netron  # noqa: E402

    assert version.parse(netron.__version__) >= version.parse(
        "6.0.2"
    ), "netron version should not less than 6.0.2"
except ImportError:

    logger.info(
        "cant import netron, please install netron: pip install netron==6.0.2"
    )
    sys.exit(1)

netron_dir = os.path.dirname(netron.__file__)
pytorch_metadata_file = os.path.join(netron_dir, "pytorch-metadata.json")
bak_metadata_file = pytorch_metadata_file + ".bak"
if not os.path.exists(bak_metadata_file):
    shutil.copyfile(pytorch_metadata_file, bak_metadata_file)
with open(bak_metadata_file, "r") as f:
    pytorch_metadata = json.load(f)
pytorch_metadata.extend(patch_hz_ops_metadata)
with open(pytorch_metadata_file, "w") as f:
    json.dump(pytorch_metadata, f, indent=2)
