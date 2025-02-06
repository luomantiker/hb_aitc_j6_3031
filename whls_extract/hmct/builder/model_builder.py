import copy
import logging
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from hmct.common import (
    Dataset,
    HbdkDictParser,
    InputDictParser,
    ModelDebugger,
    QuantConfig,
    create_batch_input_shape,
    load_quant_config_by_legacy,
)
from hmct.converter import optimize, prepare
from hmct.ir import OnnxModel, save_model
from hmct.ir.horizon_onnx import global_attributes, quant_attributes
from hmct.quantizer import calibrate, precompile, quantize
from hmct.reporter import (
    calculate_hybrid_type,
    calculate_quant_type,
    calculate_similarity,
    print_model_info,
    save_quant_info,
)
from hmct.version import __version__

if int(__version__[0]) == 1:
    from hmct.compiler import HybridBuilder, compile


class BuildInfo:
    def __init__(self, info):
        self.info = info
        logging.info(f"Start to {self.info}.")

    def __del__(self):
        logging.info(f"End to {self.info}.")


class ModelBuilder:
    """Builder for hybrid or ptq model.

    Convert input model and build hybrid or ptq model.
    Only onnx model is supported as input model now.
    """

    def __init__(
        self,
        onnx_model: OnnxModel,
        march: str,
        check_mode: bool = False,
        return_hybrid_model: bool = True,
        save_model: bool = False,
        verbose: bool = True,
        name_prefix: str = "",
        output_nodes: Optional[Sequence[str]] = None,
        debug_mode: Optional[Union[bool, Sequence[str]]] = None,
        optimization: Optional[Sequence[str]] = None,
        skip_step: Optional[Sequence[str]] = None,
        cali_dict: Optional[Dict[str, Any]] = None,
        node_dict: Optional[Dict[str, Dict[str, Any]]] = None,
        hbdk_dict: Optional[Dict[str, Union[str, Dict]]] = None,
        input_dict: Optional[Dict[str, Dict[str, Any]]] = None,
        user_quant_config: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs,
    ):
        """Initialize the model builder.

        Args:
            onnx_model: input onnx model for convert.
            march: Architecture of BPU. Available values are
                ['nash', 'bayes', 'bayes-e', 'bernoulli2', 'bernoulli']
            check_mode: True to run in check mode, False in build mode.
            return_hybrid_model: Whether return hybrid model or not.
                Return hybrid model if True, return ptq model if False.
            save_model: Whether to save the models generated during
                the convert process. True to save the models, and False to
                not save them.
            verbose: Whether to print model quant info when conversion
                is completed.
            name_prefix: The name or path prefix for saved models.
            output_nodes: This is a sequence which specifies model
                output node names.
            debug_mode: A sequence to specify debug options during
                model convert. The supported options are shown in
                horizon_nn/debug/model_debug.py
            optimization: A sequence to specify optimization options
                for model convert. These options can improve the performance or
                accuracy of the deployment model. The supported options are
                shown in onnx/onnx/quantization/optimization_configs.h
            skip_step: A sequence of steps to skip during the model
                convert. Available values are ['skip_optimizer',
                'skip_calibrater', 'skip_quantizer', 'skip_compiler']
            cali_dict: This is a dict param including calibration
                related parameters.
            node_dict: This is a dict param including node related
                parameters. Its keys are the name of affected nodes, and
                values are dict of the actions on the corresponding node.
            hbdk_dict: This is a dict param including compiler related parameters.
                All the parameters will be transparently transmitted to the compiler.
            input_dict: This is a dict param including input parameters.
                Its keys are names of input variables, and values
                are also dicts contain the paired parameters.
            user_quant_config: This is path of a json file including
                quantization related parameters.
            **kwargs: Arbitrary keyword arguments.
        """
        # parse input onnx model to convert
        self.input_model = onnx_model
        # parse BPU architecture for model deployment
        assert march in [
            "nash",
            "bayes",
            "bayes-e",
            "bernoulli2",
            "bernoulli",
            "b40",
            "experiment",
        ]
        self.march = march
        # parse check_mode to determine whether to check or build model
        self.check_mode = check_mode
        # parse returned model type to determine model convert chain
        self.return_hybrid_model = return_hybrid_model
        # parse save_model to determine whether to save generated models
        self.save_model = save_model
        # set the opset version
        if int(__version__[0]) == 1:
            self.op_convert = kwargs.get("op_convert", True)
            self.opset_version = 11
        else:
            self.op_convert = kwargs.get("op_convert", True)
            self.opset_version = kwargs.get("opset_version", 19)
        # parse verbose to determine whether to print model quant info
        self.verbose = verbose
        # parse name_prefix to determine name or path prefix for saved models
        if (
            name_prefix != ""
            and not name_prefix.endswith("/")
            and not name_prefix.endswith("_")
        ):
            name_prefix += "_"
        self.name_prefix = name_prefix
        # parse designated output nodes
        self.output_nodes = [] if output_nodes is None else list(output_nodes)
        # parse options for model debug
        self.model_debugger = ModelDebugger()
        if not isinstance(debug_mode, bool):
            debug_mode = [] if debug_mode is None else list(debug_mode)
        self.model_debugger.register_debug_methods(debug_mode)
        # parse calibration_data
        self.calibration_data = (
            None
            if cali_dict is None
            else cali_dict.get("calibration_data")
            or cali_dict.get("calibration_loader")
        )
        # parse quant relevant config
        self.is_qat = any(
            node.op_type
            in ["QuantizeLinear", "DequantizeLinear", "HzQuantize", "HzDequantize"]
            for node in self.input_model.graph.nodes
        )
        self.quant_config = QuantConfig(march)
        load_quant_config_by_legacy(
            self.quant_config, self.is_qat, cali_dict, node_dict, optimization
        )
        self.user_quant_config = copy.deepcopy(user_quant_config)
        # parse skipped stages during model convert
        skip_step = set() if skip_step is None else set(skip_step)
        if "skip_calibrater" in skip_step:
            skip_step.update(["skip_quantizer", "skip_compiler"])
        elif "skip_quantizer" in skip_step:
            skip_step.update(["skip_compiler"])
        self.skip_step = list(skip_step)
        # parse the {cali/node/hbdk/input}_dict
        self.hbdk_dict_parser = HbdkDictParser(hbdk_dict)
        self.input_dict_parser = InputDictParser(
            self.march,
            input_dict,
            self.hbdk_dict_parser.get_input_source(),
        )

        # initialize global attributes
        self.init_global_configs()

        # generated models of different stages during convert
        self.original_model = None
        self.optimized_model = None
        self.calibrated_model = None
        self.quantized_model = None
        self.ptq_model = None
        self.hybrid_model = None

        # cal type used to display in benchmark results
        self.cal_type = ""

        # model && node information after convert
        self.node_similarity_info = {}
        self.output_similarity_info = {}
        self.model_quant_type_info = {}
        self.model_hybrid_info = {}

    def init_global_configs(self):
        """Initialize some user's configs into global attributes."""
        # clear global_attrtibutes
        global_attributes.clear()
        # set march
        logging.info(
            "The specified model compilation architecture: "
            f"{self.quant_config.march}."
        )
        global_attributes.set_march(self.quant_config.march)
        # set quantization mode
        if self.is_qat:
            global_attributes.set_quantization_mode("preq")
        else:
            global_attributes.set_quantization_mode("autoq")

        # set default asymmetric mode
        global_attributes.set_asymmetric_mode("disable_all")
        # set optimization
        logging.info(
            "The specified model compilation optimization parameters: "
            f"{self.quant_config.optimization}.",
        )
        global_attributes.set_optimization(self.quant_config.optimization)

    def get_all_models(self) -> Tuple[Union[None, "ModelProto"], ...]:  # noqa: F821
        """Get all-stage models generated by the builder.

        Returns:
            The tuple of generated models.
        """
        original_model = self.original_model.proto
        optimized_model = self.optimized_model.proto
        calibrated_model = (
            self.calibrated_model
            if self.calibrated_model is None
            else self.calibrated_model.proto
        )

        if self.return_hybrid_model:
            quantized_model = (
                self.quantized_model
                if self.quantized_model is None
                else self.quantized_model.proto
            )
            hybrid_model = (
                self.hybrid_model
                if self.hybrid_model is None
                else self.hybrid_model.proto
            )
            return (
                original_model,
                optimized_model,
                calibrated_model,
                quantized_model,
                hybrid_model,
            )

        ptq_model = self.ptq_model if self.ptq_model is None else self.ptq_model.proto
        return original_model, optimized_model, calibrated_model, ptq_model

    def save(self, onnx_model: OnnxModel, name_suffix: str) -> None:
        model_name = self.name_prefix + name_suffix
        save_model(onnx_model, model_name)
        logging.info(f"Saving model to: {model_name}.")

    def prepare(self, input_model: OnnxModel) -> OnnxModel:
        """Prepare onnx model for model convert.

        Args:
            input_model: input onnx model from user.

        Returns:
            original onnx model after preparation.
        """
        _ = BuildInfo("prepare the onnx model")

        original_model = prepare(
            onnx_model=copy.deepcopy(input_model),
            input_dict_parser=self.input_dict_parser,
            op_convert=self.op_convert,
            opset_version=self.opset_version,
            output_nodes=self.output_nodes,
        )
        original_model.check_validity()

        return original_model

    def optimize(self, original_model: OnnxModel) -> OnnxModel:
        """Optimize onnx model for model convert.

        Args:
            original_model: original onnx model after preparation.

        Returns:
            optimized onnx model after optimization.
        """
        _ = BuildInfo("optimize the onnx model")

        optimized_model = optimize(original_model=copy.deepcopy(original_model))
        optimized_model.check_validity()

        return optimized_model

    def calibrate(
        self,
        optimized_model: OnnxModel,
    ) -> Tuple[OnnxModel, str, Dataset]:
        """Calibrate onnx model for model convert.

        Args:
            optimized_model: if skip_optimizer, it's original onnx
            model after preparation, else it's optimized onnx model
            after optimization.

        Returns:
            A tuple which includes:
            calibrated onnx model after calibration,
            cal_type used to display in benchmark results and
            calibration_dataset used to calibrate model.

        """
        _ = BuildInfo("calibrate the model")

        calibrated_model, cal_type, calibration_dataset = calibrate(
            optimized_model=copy.deepcopy(optimized_model),
            quant_config=self.quant_config,
            model_debugger=self.model_debugger,
            calibration_data=self.calibration_data,
        )
        calibrated_model.check_validity()

        return calibrated_model, cal_type, calibration_dataset

    def quantize(self, calibrated_model: OnnxModel) -> OnnxModel:
        """Quantize onnx model for model convert.

        Args:
            calibrated_model: calibrated onnx model after calibration.

        Returns:
            quantized onnx model after fixed-point quantization.
        """
        _ = BuildInfo("quantize the model")

        quantized_model = quantize(
            calibrated_model=copy.deepcopy(calibrated_model),
            input_dict_parser=self.input_dict_parser,
        )
        quantized_model.check_validity()

        return quantized_model

    def precompile(self, calibrated_model: OnnxModel) -> OnnxModel:
        """Precompile onnx model for hbdk4.

        Args:
            calibrated_model: calibrated onnx model after calibration.

        Returns:
            ptq onnx model after precompile for hbdk4.
        """
        _ = BuildInfo("precompile the model")

        ptq_model = precompile(
            calibrated_model=copy.deepcopy(calibrated_model),
            original_model=copy.deepcopy(self.original_model),
            batched_input_shapes=create_batch_input_shape(
                calibrated_model.graph.input_shapes,
                self.input_dict_parser.get_input_batches(),
            ),
        )
        ptq_model.check_validity()

        return ptq_model

    def compile(self, quantized_model: OnnxModel) -> Tuple[OnnxModel, "HybridBuilder"]:
        """Compile onnx model for model convert.

        Args:
            quantized_model: quantized onnx model after fixed-point quantization.

        Returns:
            hybrid onnx model after compilation.
        """
        _ = BuildInfo(f"compile the model with march {self.march}")

        hybrid_model, hybrid_builder = compile(
            quantized_model=copy.deepcopy(quantized_model),
            original_model=copy.deepcopy(self.original_model),
            march=self.march,
            hbdk_dict_parser=self.hbdk_dict_parser,
            model_debugger=self.model_debugger,
            batched_input_shapes=create_batch_input_shape(
                quantized_model.graph.input_shapes,
                self.input_dict_parser.get_input_batches(),
            ),
            name_prefix=self.name_prefix,
        )
        hybrid_model.check_validity()

        return hybrid_model, hybrid_builder

    def build(self) -> Union["ModelProto", None]:  # noqa: F821
        """Build model from onnx model.

        This function will convert onnx model to hybrid model or ptq model.

        Returns:
            hybrid model or ptq model if success, None otherwise.
        """
        # step-1. Prepare input model then obtain original model
        self.original_model = self.prepare(self.input_model)
        if self.save_model:
            self.save(
                onnx_model=self.original_model,
                name_suffix="original_float_model.onnx",
            )

        # step-2. Optimize original model and obtain optimized model
        if "skip_optimizer" in self.skip_step:
            self.optimized_model = copy.deepcopy(self.original_model)
        else:
            self.optimized_model = self.optimize(self.original_model)
            if self.save_model:
                self.save(
                    onnx_model=self.optimized_model,
                    name_suffix="optimized_float_model.onnx",
                )

        self.quant_config.load_user_quant_config(
            self.user_quant_config,
            self.optimized_model,
        )

        def subset_by_keys(
            config: Dict[str, Any], keys: Sequence[str]
        ) -> Dict[str, Any]:
            return {k: v for k, v in config.items() if k in keys}

        # Set model, op and node qtype to C++
        quant_attributes.set(
            subset_by_keys(
                self.quant_config.model_config, ["all_node_type", "model_output_type"]
            ),
            self.quant_config.op_config,
            self.quant_config.node_config,
        )
        self.quant_config.print_quant_config()

        # step-3. Calibrate optimized model then obtain calibrated model
        if "skip_calibrater" not in self.skip_step:
            self.calibrated_model, self.cal_type, self.calibration_dataset = (
                self.calibrate(
                    self.optimized_model,
                )
            )
            if self.save_model:
                self.save(
                    onnx_model=self.calibrated_model,
                    name_suffix="calibrated_model.onnx",
                )
        else:
            self.calibration_dataset = None

        # step-4. Quantize calibrated model then obtain quantized or ptq model.
        if "skip_quantizer" not in self.skip_step:
            if self.return_hybrid_model:
                self.quantized_model = self.quantize(self.calibrated_model)
                if self.save_model:
                    self.save(
                        onnx_model=self.quantized_model,
                        name_suffix="quantized_model.onnx",
                    )
            else:
                self.ptq_model = self.precompile(self.calibrated_model)
                if self.save_model:
                    self.save(onnx_model=self.ptq_model, name_suffix="ptq_model.onnx")

        # step-5 (optional). Compile quantized model then obtain hybrid_model
        if "skip_compiler" not in self.skip_step and self.return_hybrid_model:
            self.hybrid_model, hybrid_builder = self.compile(self.quantized_model)
            # save hybrid model
            if self.save_model and self.model_debugger.has_debug_method(
                "dump_all_models",
            ):
                self.save(onnx_model=self.hybrid_model, name_suffix="hybrid_model.onnx")
        else:
            hybrid_builder = None

        # step-6 collect node info && model output info and display in terminal/json/svg
        # step-6.1 collect model info
        if self.calibrated_model is not None:
            self.model_quant_type_info = calculate_quant_type(self.calibrated_model)
            if not self.check_mode:
                self.node_similarity_info, self.output_similarity_info = (
                    calculate_similarity(
                        calibrated_model=self.calibrated_model,
                        calibration_data_dict=self.calibration_dataset.get_data(),
                    )
                )
        self.model_hybrid_info = calculate_hybrid_type(
            quantized_model=self.quantized_model, hybrid_builder=hybrid_builder
        )
        # step-6.2 print model info
        if self.verbose:
            print_model_info(
                self.node_similarity_info,
                self.output_similarity_info,
                self.model_quant_type_info,
                self.model_hybrid_info,
            )
        # step-6.3 export to json file
        if self.save_model:
            save_quant_info(
                self.model_quant_type_info,
                self.node_similarity_info,
                self.output_similarity_info,
                self.name_prefix + "quant_info.json",
            )

        if self.return_hybrid_model:
            return (
                self.hybrid_model
                if self.hybrid_model is None
                else self.hybrid_model.proto
            )

        return self.ptq_model if self.ptq_model is None else self.ptq_model.proto
