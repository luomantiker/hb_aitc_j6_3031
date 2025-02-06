"""align bpu validation tools, Only support int-infer."""

import argparse
from functools import partial

import horizon_plugin_pytorch as horizon
import torch
import torchvision

from hat.registry import build_from_registry
from hat.utils.config import Config
from hat.utils.logger import init_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="train config file path",
    )
    parser.add_argument(
        "--model-inputs", type=str, default=None, help="model input"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="save path for visualize output.",
    )
    parser.add_argument(
        "--use-dataset",
        action="store_true",
        help="Whether check the mlir model forward with example input.",
    )
    return parser.parse_args()


def defualt_prepare_inputs(infer_inputs):

    return [infer_inputs]


def _analyze_inputs(inputs):

    inputs_list = inputs.split(",")
    model_inputs = {}
    for each_input in inputs_list:
        name, values = each_input.split(":")
        model_inputs[name.strip()] = values.strip()

    return model_inputs


if __name__ == "__main__":
    args = parse_args()

    config = Config.fromfile(args.config)
    init_logger(f".hat_logs/{config.task_name}_infer_viz")
    horizon.march.set_march(config.get("march"))

    infer_cfg = config.get("infer_cfg")

    # get model inputs
    if args.model_inputs is not None:
        input_path = args.model_inputs
    else:
        input_path = infer_cfg.get("input_path")
        if args.use_dataset:
            gen_inputs_cfg = infer_cfg.get("gen_inputs_cfg")
            assert (
                gen_inputs_cfg is not None
            ), "You must set gen_inputs_cfg in infer_cfg when use dataset."
            dataset = build_from_registry(gen_inputs_cfg["dataset"])
            sample_idx = gen_inputs_cfg["sample_idx"]
            if len(sample_idx) == 1:
                sample_data = dataset[sample_idx[0]]
            else:
                sample_data = []
                for idx_ in range(sample_idx[0], sample_idx[1]):
                    sample_data.append(dataset[idx_])
            inputs_save_func = gen_inputs_cfg.get("inputs_save_func")
            inputs_save_func(sample_data, input_path)

    prepare_inputs = infer_cfg.get("prepare_inputs", defualt_prepare_inputs)
    prepared_inputs = prepare_inputs(input_path)

    # build data transforms
    transforms = infer_cfg.get("transforms", None)

    if transforms is not None:
        transforms = build_from_registry(transforms)
        transforms = torchvision.transforms.Compose(transforms)

    # build model and load ckpt
    model = build_from_registry(infer_cfg.get("model"))
    model.eval()

    viz_func = build_from_registry(infer_cfg.get("viz_func"))
    if isinstance(viz_func, list):
        for i, f in enumerate(viz_func):
            viz_func[i] = partial(f, save_path=args.save_path)
    else:
        viz_func = partial(viz_func, save_path=args.save_path)

    process_inputs = infer_cfg.get("process_inputs")
    process_outputs = infer_cfg.get("process_outputs")

    for prepared_input in prepared_inputs:
        model_input, vis_inputs = process_inputs(prepared_input, transforms)
        with torch.no_grad():
            model_outputs = model(model_input)

        outputs = process_outputs(model_outputs, viz_func, vis_inputs)
        if outputs is not None:
            print(outputs)
