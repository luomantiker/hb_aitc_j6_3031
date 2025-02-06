import os
from abc import abstractmethod
from numbers import Real
from typing import Callable, Optional, Tuple, Type, Union

import torch

from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import require_packages

try:
    import horizon_plugin_profiler as quant_profiler
except ImportError:
    from horizon_plugin_pytorch.utils import quant_profiler


__all__ = [
    "FeaturemapSimilarity",
    "ProfileFeaturemap",
    "CheckShared",
    "CheckFused",
    "CheckQConfig",
    "CompareWeights",
    "CheckDeployDevice",
    "ModelProfiler",
    "ModelProfilerv2",
    "HbirModelProfiler",
]


@OBJECT_REGISTRY.register
class BaseModelProfiler(object):
    """Base class for defining the process of model analysis."""

    @require_packages("horizon_plugin_pytorch>=1.1.2")
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, model):
        raise NotImplementedError


@OBJECT_REGISTRY.register
class FeaturemapSimilarity(BaseModelProfiler):
    """Compute the similarity of two models.

    Compute the similarity of feature maps. The input models can be float/
    fused/calibration/qat/quantized model.

    Args:
        similarity_func: similarity computation function. Support "Cosine",
            "MSE", "L1", "KL", "SQNR", or any user-defined Callable object. If
            it is a user-defined object, it should return a scalar or tensor
            with only one number. Otherwise the result shown may be unexpected.
            Default: "Cosine"
        threshold: if similarity value exceeds or less than this threshold,
            the featuremap name will be shown in red color.If threshold is
            none, it will be set to different values according to different
            similarity functions. Default: None
        devices: run model on which devices (cpu, gpu). If can be \
            - None. Run model with given inputs \
            - torch.device. Both models and given inputs will be moved on \
                this specified device \
            - tuple. A tuple of 2 torch.devices. The two models will be moved \
                on specified devices seperatedly. It may be used to compare the
                CPU and GPU results difference.
        out_dir: path to save the result txt and picture. If None, will save in
            the current directory. Default: None

    Returns:
        A List of list. Each list is each layer similarity info in format
        [index, module name, module type, similarity, scale, atol,
        atol(N scale), single op error(N scale)]
    """

    def __init__(
        self,
        similarity_func: Union[str, callable] = "Cosine",
        threshold: Optional[Real] = None,
        devices: Union[torch.device, tuple, None] = None,
        out_dir: Optional[str] = None,
    ):
        super(FeaturemapSimilarity, self).__init__()
        self.similarity_func = similarity_func
        self.threshold = threshold
        self.devices = devices
        self.out_dir = out_dir

    def __call__(self, model1, model2, inputs):
        return quant_profiler.featuremap_similarity(
            model1,
            model2,
            inputs,
            self.similarity_func,
            self.threshold,
            self.devices,
            self.out_dir,
        )


@OBJECT_REGISTRY.register
class ProfileFeaturemap(BaseModelProfiler):
    """Profile featuremap value with log or tensorboard.

    Print min/max/mean/var/scale of each feature profiled by `get_raw_features`
    by default. If `with_tensorboard` set True, the histogram of each feature
    will be shown in tensorboard, which is useful to see the data distribution.

    If you want to get more info about features, you can define your custom
    profile functions to process the results of `get_raw_features`.

    Args:
        prefixes: get features info by the prefix of qualified name
            Default: tuple().
        types: get features info by module type. Default: tuple().
        device: model run on which device. Default: None
        preserve_int: if True, record each op result in int type.
            Default: False
        use_class_name: if True, record class name not class type.
            Default: False
        skip_identity: if True, will not record the result of Identity module.
            Default: False
        with_tensorboard: whether to use tensorboard. Default: False
        tensorboard_dir: tensorboard log file path. Default: None
        print_per_channel_scale: whether to print per channel scales.
            Default: False
        show_per_channel: show each featuremap in per channel ways
            in tensorboard. Default: False
        out_dir: path to save the result txt and picture. If None, will save in
            the current directory. Default: None
        file_name: result file name. If None, will save result and fig with
            name 'statistic'.(statistic.txt and statistic.html). Default: None
        profile_func(callable, None): you custom featuremap profiler function.
            Default: None

    Returns:
        A List of list. Each list is each layer statistic in format
        [index, module name, module type, attr, min, max, mean, var, scale]
    """

    def __init__(
        self,
        # get_raw_features args
        prefixes: Tuple = (),
        types: Tuple = (),
        device: torch.device = None,
        preserve_int: bool = False,
        use_class_name: bool = False,
        skip_identity: bool = False,
        # profile_featuremap args
        with_tensorboard: bool = False,
        tensorboard_dir: Optional[str] = None,
        print_per_channel_scale: bool = False,
        show_per_channel: bool = False,
        out_dir: Optional[str] = None,
        file_name: Optional[str] = None,
        profile_func: Optional[callable] = None,
    ):
        super(ProfileFeaturemap, self).__init__()
        # get_raw_features args
        self.prefixes = prefixes
        self.types = types
        self.device = device
        self.preserve_int = preserve_int
        self.use_class_name = use_class_name
        self.skip_identity = skip_identity
        # profile_featuremap args
        self.with_tensorboard = with_tensorboard
        self.tensorboard_dir = tensorboard_dir
        self.print_per_channel_scale = print_per_channel_scale
        self.show_per_channel = show_per_channel
        self.out_dir = out_dir
        self.file_name = file_name
        self.profile_func = profile_func

    def __call__(self, model, inputs):
        info = quant_profiler.get_raw_features(
            model,
            inputs,
            self.prefixes,
            self.types,
            self.device,
            self.preserve_int,
            self.use_class_name,
            self.skip_identity,
        )
        if self.profile_func is None:
            return quant_profiler.profile_featuremap(
                info,
                self.with_tensorboard,
                self.tensorboard_dir,
                self.print_per_channel_scale,
                self.show_per_channel,
                self.out_dir,
                self.file_name,
            )
        else:
            return self.profile_func(info)


@OBJECT_REGISTRY.register
class CheckShared(BaseModelProfiler):
    """Checking if model has shared ops.

    Count called times for all leaf modules in a model.

    Args:
        check_leaf_module (callable, optional): A function to check if
            a module is leaf. Pass None to use pre-defined `is_leaf_module`.
            Default: None.
        print_tabulate (bool, optional): Whether print the result as tabulate.
            Default: True.

    Returns:
        Dict[str, int]:
            The qualified name and called times of each leaf module.
    """

    def __init__(
        self,
        check_leaf_module: Optional[callable] = None,
        print_tabulate: bool = True,
    ):
        super(CheckShared, self).__init__()
        self.check_leaf_module = check_leaf_module
        self.print_tabulate = print_tabulate

    def __call__(self, model, inputs):
        return quant_profiler.get_module_called_count(
            model,
            inputs,
            self.check_leaf_module,
            self.print_tabulate,
        )


@OBJECT_REGISTRY.register
class CheckFused(BaseModelProfiler):
    """Checking if model has unfused ops.

    Check unfused modules in a model.
    NOTE: This function is only capable to check unfused modules. For the
    correctness of fusion, please use `featuremap_similarity` to compare
    the feature between fused and unfused model.

    Args:
        print_tabulate (bool): Whether print the result as tabulate.
            Default: True.

    Returns:
        List[List[str]]:
            The qualified name of modules that can be fused.
    """

    def __init__(self, print_tabulate: bool = True):
        super(CheckFused, self).__init__()
        self.print_tabulate = print_tabulate

    def __call__(self, model, inputs):
        return quant_profiler.check_unfused_operations(
            model, inputs, self.print_tabulate
        )


@OBJECT_REGISTRY.register
class CheckQConfig(BaseModelProfiler):
    """Check quantization configuration of qat model.

    This function
    1) checks activation and weight quantization configurations of each layer
        in the model. These infos will be saved in "qconfig_info.txt".
    2) checks input and output dtype of each layer in the model.

    Defaultly this function prints warnings when checking:
    1) activation = None
    2) fixed scale observer
    3) not qint8 weight
    4) module inputs and outputs dtype diff
    If you want to check more info, define a custom check function and use
    `custom_check_func` parameter.

    Args:
        prefixes: get features info by the prefix of qualified name
            Default: tuple().
        types: get features info by module type. Default: tuple().
        custom_check_func: a user-defined function to check other infos. This
            function is invoked in module hooks, so it has the same signature
            with torch.nn.Module hooks:
                func(module, input, output) -> None
        out_dir: path to save the result txt 'qconfig_info.txt'. If None, will
            save in the current directory. Default: None

    Returns:
        (out_info_list, weight_info_list, warning_layers_info)
    """

    def __init__(
        self,
        prefixes: Tuple = (),
        types: Tuple = (),
        custom_check_func: Optional[Callable] = None,
        out_dir: Optional[str] = None,
    ):
        super(CheckQConfig, self).__init__()
        self.prefixes = prefixes
        self.types = types
        self.custom_check_func = custom_check_func
        self.out_dir = out_dir

    def __call__(self, model, inputs):
        return quant_profiler.check_qconfig(
            model,
            inputs,
            self.prefixes,
            self.types,
            self.custom_check_func,
            self.out_dir,
        )


@OBJECT_REGISTRY.register
class CompareWeights(BaseModelProfiler):
    """Compare weights of float/qat/quantized models.

    This function compares weights of each layer based on
    torch.quantization._numeric_suite.compare_weights. The weight similarity
    and atol will be print on the screen and save in "weight_comparison.txt".
    If you want to see histogram of weights, set with_tensorboard=True.

    Args:
        similarity_func: similarity computation function. Support "Cosine",
            "MSE", "L1", "KL", "SQNR" or any user-defined Callable object. If
            it is a user-defined object, it should return a scalar or tensor
            with only one number. Otherwise the result shown may be unexpected.
            Default: "Cosine"
        with_tensorboard: whether to use tensorboard. Default: False
        tensorboard_dir: tensorboard log file path. Default: None
        out_dir: path to save the result txt and picture. If None, will save in
            the current directory. Default: None

    Returns:
        A weight comparison dict with schema:
            * KEY (str): module name (Eg. layer1.0.conv.weight)
            * VALUE (dict): a dict of the corresponding weights in two models:
                "float": weight value in float model
                "quantized": weight value in qat/quantized model

        A list of list. Each list is each layer weight similarity in format
        [module name, similarity, atol(N scale)]
    """

    def __init__(
        self,
        similarity_func="Cosine",
        with_tensorboard: bool = False,
        tensorboard_dir: Optional[str] = None,
        out_dir: Optional[str] = None,
    ):
        super(CompareWeights, self).__init__()
        self.similarity_func = similarity_func
        self.with_tensorboard = with_tensorboard
        self.tensorboard_dir = tensorboard_dir
        self.out_dir = out_dir

    def __call__(self, float_model, qat_quantized_model):
        return quant_profiler.compare_weights(
            float_model,
            qat_quantized_model,
            self.similarity_func,
            self.with_tensorboard,
            self.tensorboard_dir,
            self.out_dir,
        )


@OBJECT_REGISTRY.register
class CheckDeployDevice(BaseModelProfiler):
    """Check deploy device(BPU or CPU) of hybrid model.

    Args:
        print_tabulate (bool, optional): Whether print the result as tabulate.
            Defaults to True.
        out_dir: path to save the result txt 'deploy_device.txt'. If None, will
            save in the current directory. Default: None

    Returns:
        A dict of model deploy infos with schema
            * KEY (str): module name
            * VALUE (Tuple): (deploy device(BPU or CPU), module type)
    """

    def __init__(
        self, print_tabulate: bool = True, out_dir: Optional[str] = None
    ):
        super(CheckDeployDevice, self).__init__()
        self.print_tabulate = print_tabulate
        self.out_dir = out_dir

    def __call__(self, model):
        return quant_profiler.check_deploy_device(
            model, self.print_tabulate, self.out_dir
        )


@OBJECT_REGISTRY.register
class ModelProfiler(BaseModelProfiler):
    """Profiler the models using debug tools and show result in one page.

    This function computes
    1) similarity, statistics, weights similarity and shared ops of the given
        models
    2) check unfused ops of the float model and qconfig of the qat model, which
        controlled by `mode`
    The results are shown in one html page named `profiler.html`, which stored
    in default dir or `out_dir`.

    NOTE:
        1) Only support models compared in any two adjacent stages.
            `float vs qat` or `qat vs quantized` is supported, while
            `float vs quantized` or `qat vs float` is unsupported.
        2) Visual model structures in onnx format and featuremap histogram are
            not shown in the html file. You can call `export_to_onnx/
            export_quantized_onnx` and `profile_featuremap` with
            `with_tensorboard=True`. Custom args can also be passed by
            `kwargs_dict`.

    Args:
        example_inputs: model inputs
        mode: specific which two models to be compared. Only three modes shown
            below are supported
            "FvsQ": float vs qat. In this mode, `model2` can be either
                calibration or qat model.
            "QvsQ": qat vs quantized.
            "CvsQ": calibration vs qat.
        out_dir: path to save `profiler.html` and all other result files. If
            None, results are saved in `horizon_quant_debug` dir in current dir
        kwargs_dict: kwargs of debug tools functions in dict format. E.g.
            kwargs_dict = {
                "FeaturemapSimilarity": {
                    "similarity_func": Cosine,
                },
                "ProfileFeaturemap": {
                    "with_tensorboard": True,
                }
                ...
            }
            Only support 6 keys, which are the names of the other 6 debug
            profiler classes. The supported keys are:
                1) FeaturemapSimilarity
                2) ProfileFeaturemap
                3) CheckShared
                4) CheckFused
                5) CompareWeights
                6) CheckQConfig
            NOTE:
                1) model and example_inputs must not be defined in kwargs
                2) `out_dir` in kwargs will be replaced with `out_dir` in
                    `model_profiler` args
    """

    def __init__(
        self,
        mode: str,
        out_dir: Optional[str] = None,
        kwargs_dict: Optional[dict] = None,
    ):
        super(ModelProfiler, self).__init__()
        self.mode = mode
        self.out_dir = out_dir
        self.kwargs_dict = {} if kwargs_dict is not None else None
        _key_map = {
            "FeaturemapSimilarity": "featuremap_similarity",
            "ProfileFeaturemap": "profile_featuremap",
            "CheckShared": "get_module_called_count",
            "CheckFused": "check_unfused_operations",
            "CheckQConfig": "check_qconfig",
            "CompareWeights": "compare_weights",
        }
        # kwargs_dict preprocess
        if kwargs_dict is not None:
            for k, d in kwargs_dict.items():
                if k not in _key_map.keys():
                    raise KeyError(f"unknown key {k}")
                if k == "ProfileFeaturemap":
                    get_raw_features_dict = {
                        "prefixes": (),
                        "types": (),
                        "device": None,
                        "preserve_int": False,
                        "use_class_name": False,
                        "skip_identity": False,
                    }
                    for key in get_raw_features_dict.keys():
                        v = d.pop(key, None)
                        if v is not None:
                            get_raw_features_dict.update({key: v})
                    self.kwargs_dict.update({_key_map[k]: d})
                    self.kwargs_dict.update(
                        {"get_raw_features": get_raw_features_dict}
                    )
                else:
                    self.kwargs_dict.update({_key_map[k]: d})

    def __call__(self, model1, model2, inputs):
        return quant_profiler.model_profiler(
            model1,
            model2,
            inputs,
            self.mode,
            self.out_dir,
            self.kwargs_dict,
        )


@OBJECT_REGISTRY.register
class ModelProfilerv2(BaseModelProfiler):
    """Run model and save each op info.

    This function runs model and save each op info on disk, which can be show
    in a table or in tensorboard.

    Args:
        show_table: whether show each op info in a table, which will also be
            saved in statistic.txt
        show_tensorboard: whether show each op histogram in tensorboard.
        prefixes: only show ops with the prefix of qualified name
        types: only show ops with given types
        with_stack: whether show op location in code
        force_per_channel: whether show data in per channel in tensorboard
        out_dir: dir to save op infos and result files
    """

    def __init__(
        self,
        show_table: bool = True,
        show_tensorboard: bool = False,
        prefixes: Tuple[str, ...] = None,
        types: Tuple[Type, ...] = None,
        with_stack: bool = False,
        force_per_channel: bool = False,
        out_dir: Optional[str] = None,
    ):
        super(ModelProfilerv2, self).__init__()
        self.show_table = show_table
        self.show_tensorboard = show_tensorboard
        self.prefixes = prefixes
        self.types = types
        self.with_stack = with_stack
        self.force_per_channel = force_per_channel
        self.out_dir = out_dir

    def run(self, model, example_inputs, out_dir):
        with quant_profiler.ModelProfiler(model, out_dir) as profiler:
            model(example_inputs)

        return profiler

    @require_packages("horizon_plugin_profiler>=2.1.6")
    def __call__(self, model, example_inputs):
        info_path = os.path.join(self.out_dir, "op_infos")
        os.makedirs(info_path, exist_ok=True)

        profiler = self.run(model, example_inputs, info_path)

        if self.show_table:
            profiler.get_info_manager().table(
                self.out_dir,
                self.prefixes,
                self.types,
                self.with_stack,
            )

        if self.show_tensorboard:
            profiler.get_info_manager().tensorboard(
                self.out_dir,
                self.prefixes,
                self.types,
                self.force_per_channel,
            )


@OBJECT_REGISTRY.register
class HbirModelProfiler(ModelProfilerv2):
    """Run hbir model and save each op info.

    This function runs hbir model and save each op info on disk, which can be
    show in a table or in tensorboard.

    Args:
        show_table: whether show each op info in a table, which will also be
            saved in statistic.txt
        show_tensorboard: whether show each op histogram in tensorboard.
        prefixes: only show ops with the prefix of qualified name
        types: only show ops with given types
        with_stack: whether show op location in code
        force_per_channel: whether show data in per channel in tensorboard
        out_dir: dir to save op infos and result files
    """

    def __init__(
        self,
        show_table: bool = True,
        show_tensorboard: bool = False,
        prefixes: Tuple[str, ...] = None,
        types: Tuple[Type, ...] = None,
        with_stack: bool = False,
        force_per_channel: bool = False,
        out_dir: Optional[str] = None,
    ):
        super(HbirModelProfiler, self).__init__(
            show_table,
            show_tensorboard,
            prefixes,
            types,
            with_stack,
            force_per_channel,
            out_dir,
        )

    def run(self, model, example_inputs, out_dir):
        from horizon_plugin_profiler.hbir_model_profiler import (
            HbirModelProfiler,
        )
        from horizon_plugin_profiler.utils.model_helper import (
            HbirModuleWrapper,
        )

        model = HbirModuleWrapper(model)
        with HbirModelProfiler(model, out_dir) as profiler:
            model(example_inputs)

        return profiler
