# HBDK Tool API Reference

## hbdk api introduction

General restrictions:

The maximum number of model inputs/outputs should not exceed 256.

The dimensions of all tensors in the model should not exceed 10.

The name of the input and output of the model must be unique.

The data types we support are ui8/si8/si16/ui32/si32/si64/float/bool. For specific data types supported by a certain operator, please refer to the operator constraint document.

### Module: hbdk4.compiler.onnx

#### `export`(proto onnx.ModelProto, *, name Optional[str] = None) -> Module
```python
export an onnx module to hbir mlir

    Args:
        proto (onnx.ModelProto): onnx protobuf
        name (Optional[str], optional): rename the onnx function. "None" means using onnx graph name

    Returns:
        Module: a helper for mlir.Module that manages hbdk operations
```

#### `statistics`(proto onnx.ModelProto)
```python
Print op statics of given onnx module.

    Args:
        proto (onnx.ModelProto): onnx protobuf
```

### Module: hbdk4.compiler.torch

#### `export`( jit torch.jit.ScriptModule, example_input Any, *, name Optional[str] = None, input_names List[str] = None, output_names List[str] = None, lower_non_tensor bool = True) -> Module
```python
export a torch.jit.ScriptModule to hbir mlir

    Args:
        jit (torch.jit.ScriptModule): a ScriptModule created from torch.jit.trace
        example_input (Any): input format of the ScriptModule, used for analysis
        name (Optional[str], optional): rename the function. "None" means uses the name recorded in ScriptModule.
        input_names (List[str], optional): rename inputs. "None" means uses input names recorded in ScriptModule.
        output_names (List[str], optional): rename outputs. "None" means uses input names recorded in ScriptModule.
        lower_non_tensor (bool, optional): flatten the pytree in ScriptModule input and return or keep the tree in hbir.

    Returns:
        Module: a helper for mlir.Module that manages hbdk operations
```

#### `statistics`(jit torch.jit.ScriptModule, example_input Any)
```python
Print op statics of given ScriptModule module.

    Args:
        jit (torch.jit.ScriptModule): a ScriptModule created from torch.jit.trace
```

### Module: hbdk4.compiler.apis

#### `load`(path str) -> Module
```python
load mlir text or bytecode to mlir.Module

    Args:
        * path (str): A filesystem path to load bytecode ended with ".bc"

    Raises:
        * ValueError: When "path" is not ended with ".bc"

    Returns:
        * Module: a helper for mlir.Module that manages hbdk operations
```

#### `save`(m Module, path str) -> None
```python
save mlir.Module to mlir bytecode

    Args:
        * m (Module): a helper for mlir.Module that manages hbdk operations
        * path (str): A filesystem path to save bytecode. Must end with ".bc"

    Raises:
        * ValueError: When "path" is not ended with ".bc"
```

#### `convert`( m Module, march Union[MarchBase, str], advice=False, advice_path="", **kwargs) -> Module
```python
Convert hbir to backend ir.

    Args:
        * m (Module): a helper for mlir.Module that manages hbdk operations
        * march (Union[MarchBase, str]): BPU march, options are "bayes", "nash-e", "nash-m", "nash-p"
        * advice (bool, optional): Set whether to enable op check
        * advice_path (str, optional): path to store op check info. Defaults to empty, print op check info directly without saving the file
```

#### `statistics`(m Module) -> list
```python
Print op statics of given mlir module.

    Args:
        * m (Module): a helper for mlir.Module that manages hbdk operations
```

#### `link`(hbo_list List[Hbo], output_path str, desc Optional[str] = None)
```python
Link hbo to hbm

    Args:
        * hbo_list (List[Hbo): A List of Hbo, which is general by compile
        * output_path (str, required): A filesystem path to save hbm. Must ends with ".hbm"
        * desc (str optional): Generate a description of hbm when linking, if this parameter is not given, the description information of hbm will not be generated
```

#### `compile`( m Module, path str, march Union[MarchBase, str], opt int = 2, jobs int = 4, max_time_per_fc float = 0.0, debug bool = False, hbdk3_compatible_mode bool = False, progress_bar bool = False, advice float = 0.0, balance int = 100, input_no_padding bool = False, output_no_padding bool = False) -> Union[Hbm, Hbo]
```python
Compile hbir module to hbm or hbo.

    If the suffix of the output is 'hbo', it will compile to generate an hbo file.
    Afterward, you need to call the link function to perform linking or packing operations.

    Args:
        * m (Module): a helper for mlir.Module that manages hbdk operations
        * march (Union[MarchBase, str]): BPU march, options are "bayes", "nash-e", "nash-m", "nash-p"
        * path (str, required): A filesystem path to save hbm or hbo. Must ends with ".hbm" or ".hbo"
        * opt (int, optional): Optimization level. Defaults to 2.
        * jobs (int, optional): Number of threads launched during compiler optimization. Defaults to 4.
        * max_time_per_fc (float, optional): Set maximum time constraint (unit:us) for per funccall.
        * debug (bool, optional): Set whether to contain debug info in HBM
        * hbdk3_compatible_mode (bool, optional): Set whether to compile in hbdk3 compatible mode, True use hbm3 and False use hbm4
        * progress_bar(bool, optional): Set whether to show progress bar
        * advice(float, optional): Print advice on ops that take longer more than the specified time (unit:us)
        * balance(int, optional): Specify a integer (recommend 2) to balance cycles and DDR access.
                                The integer should be between 0 (minimal DDR access) and 100 (minimal cycles)
        * input_no_padding (bool, optional): Set whether model input is native without padding
        * output_no_padding (bool, optional): Set whether model output is native without padding

    Raises:
        * ValueError: When "path" is not ended with ".hbm"
        * ValueError: When "balance" is not in range of [0,100]
```

#### `visualize`( m Module, onnx_file Optional[str] = None, use_netron Optional[bool] = False)
```python
Generate visualizable mlir onnx and view it in netron.

    Args:
        * m (Module): a helper for mlir.Module that manages hbdk operations
        * onnx_file (str): path to store onnx proto if it is None then create directory in /tmp
        * use_netron (bool): if True, start netron server to view onnx, otherwise just generate onnx

```

### Module: hbdk4.compiler.hbm_tools

#### `hbm_extract_desc`(model str) -> dict
```python
Extract hbm desc info

    DEPRECATED: It will be removed in the future

    Args:
        * model (str): Hbm model file name

    Return:
        * desc_dict (dict): Hbm desc info
```

#### `hbm_update_desc`(model str, desc_dict dict)
```python
Update hbm desc info

    DEPRECATED: It will be removed in the future

    Args:
        * model (str): Hbm model file name
        * desc_dict (dict): Hbm desc info
```

#### `hbm_perf`(model str, output_dir str = None)
```python
HBM performance analysis

    Args:
        * model (str): Hbm model file name
        * output_dir (str): Output directory to hold the results, default to the current path

    Return:
        * 0 if no error
```

### Class: hbdk4.compiler.overlay.Argument

#### `is_removable`(self) -> Tuple
```python
Check if the attached operation is removable. The operation should be single input and single output. For input argument, it should only be used by the operation. For output argument, the operation input should only be used by the operation.

    Returns:
        Tuple: The first element is bool indicating the removable flag. The second element is the diagnostic if it cannot be removed.
```

#### `get_attached_op`(self) -> List[Operation]
```python
Get the argument attached operations. For input argument, return operations uses the argument. For output argument, return the operation defining the argument.

    Returns:
        List[Operation]: _description_
```

#### `remove_attached_op`(self)
```python
Remove the only attached operation

    Returns:
        Tuple: The first element is True when the removal done. The second element is the diagnostic if it cannot be removed.

    Note:
        Quantize and Dequantize op should be removed after convert
```

#### `erase`(self)
```python
Remove the argument from function argument

    Returns:
        Tuple: The first element is True when the removal done. The second element is the diagnostic if it cannot be removed.
```

#### `insert_transpose`(self, permutes List[int])
```python
Insert transpose op. Change input/output parameter dimension order.

    Args:
        * permutes (List): Dimension transformation arrangement. Must contain all dimensions of the original input parameters, starting from 0
    Returns:
        List of newly inserted function arguments which is also the inputs/outputs of inserted transpose op

    Raises:
        ValueError when this argument is no longer valid

    Note:
        To avoid the new insertion operator not running in some conversion passes, it is recommended to call the insert_xxx api before the convert stage

    Example:
        module = load("model.bc")
        func = module[0]
        res = func.inputs[0].insert_transpose([3, 1, 2, 0])
```

#### `insert_rle`(self)
```python
Insert rle op. Run length encode on output.

    Returns:
        List of newly inserted function arguments which is the outputs of inserted rle op.

    Raises:
        ValueError when this argument is no longer valid.

    Note:
        The insert_rle api needs to be called after convert.
        If the output is dequantize op, dequantize op should be removed and then call insert_rle.

    Example:
        module = load("model.bc")
        func = module[0]
        res = func.inputs[0].insert_rle()
```

#### `insert_image_convert`(self, mode str = "nv12")
```python
Insert image_convert op. Change input parameter type.

    Args:
        * mode (str): Specify conversion mode, optional values are "nv12"(default) and "gray".

    Returns:
        List of newly inserted function arguments which is also the inputs of inserted image convert op

    Raises:
        ValueError when this argument is no longer valid

    Note:
        To avoid the new insertion operator not running in some conversion passes, it is recommended to call the insert_xxx api before the convert stage

    Example:
        module = load("model.bc")
        func = module[0]
        res = func.inputs[0].insert_image_convert("nv12")
```

#### `insert_image_preprocess`( self, mode str, divisor int, mean List[float], std List[float], is_signed bool = True)
```python
Insert image_convert op. Change input parameter type.

    Args:
        * mode (str): Specify conversion mode, optional values are "skip"(default, same as None), "yuvbt601full2rgb", "yuvbt601full2bgr", "yuvbt601video2rgb" and "yuvbt601video2bgr".

    Returns:
        List of newly inserted function arguments which is also the inputs of inserted image preprocess op

    Raises:
        ValueError when this argument is no longer valid

    Note:
        To avoid the new insertion operator not running in some conversion passes, it is recommended to call the insert_xxx api before the convert stage

    Example:
        module = load("model.bc")
        func = module[0]
        res = func.inputs[0].insert_image_preprocess("yuvbt601full2rgb", 255, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], True)
```

#### `insert_roi_resize`( self, mode str, interp_mode="bilinear", pad_mode="constant", pad_value Optional[tuple] = (0, -128))
```python
Insert roi_resize op. Change input parameter type.

    Args:
        * mode (str): Specify conversion mode, optional values are "nv12" and "gray".
        * interp_mode (str): Specify interpolation mode, optional values are "bilinear"(default) and "nearest".
        * pad_mode (str): Specify fill mode, optional values are "constant"(default) and "border".
        * pad_value (tuple): Specify the padding value for Y and UV in custom pad_mode, default values are (0, -128).

    Returns:
        List of newly inserted function arguments which is also the inputs of inserted roi resize op

    Raises:
        ValueError when this argument is no longer valid

    Note:
        To avoid the new insertion operator not running in some conversion passes, it is recommended to call the insert_xxx api before the convert stage

    Example:
        module = load("model.bc")
        func = module[0]
        res = func.inputs[0].insert_roi_resize(
            mode = "nv12",
            interp_mode = "nearest",
            pad_mode = "constant",
            pad_value = (66, 77)
        )
```

#### `insert_split`(self, dim int)
```python
Insert split op.
    Split a single input/output parameter into multiple input/output parameters with a specified dimension of 1.

    Args:
        * dim (int): Dimension along which to split the tensor.

    Returns:
        List of newly inserted function arguments which is also the inputs/outputs of inserted concat/slice op

    Note:
        To avoid the new insertion operator not running in some conversion passes, it is recommended to call the insert_xxx api before the convert stage

    Raises:
        ValueError when this argument is no longer valid


    Example:
        module = load("model.bc")
        func = module[0]
        res = func.inputs[0].insert_split(0)
```

### Class: hbdk4.compiler.overlay.Module

#### `functions`(self) -> List[Function]
```python
return all functions in module

    Returns:
        List[FunctionHelper]: function are wrapped in FunctionHelper in pair with its symbol name
```

#### `graphs`(self) -> List[Function]
```python
return all functions in module

    Returns:
        List[FunctionHelper]: function are wrapped in FunctionHelper in pair with its symbol name
```

### Class: hbdk4.compiler.overlay.Function

#### `remove_io_op`(self, op_types=None, op_names=None)
```python
Experimental function to remove nodes from the model based on types or names

    Note:
        Quantize and Dequantize op should be removed after convert

    Args:
        op_types(list[str]|tuple[str]): a list/tuple of types to remove
        op_names(list[str]|tuple[str]): a list/tuple of names to remove

    Example:
        module = load("model.bc")
        func = module[0]
        func.remove_io_op(['Dequantize','Transpose','Cast'])
```
## hbdk api example usage

### export model and view model information
#### onnx
```python
import onnx

onnx_model = onnx.load(onnx_path)
from hbdk4.compiler.onnx import statistics

# print onnx op list and quantity
statistics(onnx_model)

from hbdk4.compiler.onnx import export

# export onnx to hbir mlir
# the shape must have already been recorded in onnx proto
# specify the model name by name, and if default, extract it from onnx proto
exported_module = export(onnx_model, name="OnnxModel")
print("export onnx to hbir mlir successfully")

from hbdk4.compiler import statistics, visualize

# print hbir op list and quantity
statistics(exported_module)
# generate visual model Onnx but do not generate view netron link
visualize(exported_module)
# generate a netron link to open a browser and view the model
visualize(exported_module, use_netron=True)
```

### serialize model
```python
import os
from hbdk4.compiler import save, load

# serialize mlir.Module to bytecode
save(exported_module, "converted.bc")
if os.path.exists("converted.bc"):
    print("save mlir.Module to bytecode successfully")

# deserialize bytecode to mlir.Module
restored_module = load("converted.bc")
print("load bytecode to mlir.Module successfully")
```

### fixed point model
```python
from hbdk4.compiler import convert, March

# convert hbir to backend ir
converted_module = convert(exported_module, March.nash_e, advice=False)
print("convert hbir to backend ir successfully")

# when advice=True, the following op check info will be output
# op check info like this
"""
  {
    "backend": "bpu",
    "loc": "loc(fused<#hbdk.track<resultMap = [(d0, d1) -> (d0, d1)], resultType = [!qnt.uniform<si8:f32, 0.0024351498577743769>]>>[unknown])",
    "op": "%12 = \"hbir.reshape\"(%11) <{foldable = true, shape = [12000, 64]}> : (tensor<1x12000x1x64xsi8>) -> tensor<12000x64xsi8>",
    "tensor_names": [
      {
        "data_type": "si8",
        "tensor_name": ""
      }
    ]
  },
  {
    "backend": "external_cpu",
    "choice_reason": "hbir.point_pillar_scatter is not currently supported on bpu.hbir.point_pillar_scatter is not currently supported on vpu.",
    "loc": "loc(fused<#hbdk.track<resultType = [!qnt.uniform<si8:f32, 0.0024351498577743769>]>>[unknown])",
    "op": "%13 = \"hbir.point_pillar_scatter\"(%12, %coords) <{outShape = [1, 432, 496, 64]}> : (tensor<12000x64xsi8>, tensor<12000x4xsi32>) -> tensor<1x432x496x64xsi8>",
    "tensor_names": [
      {
        "data_type": "si8",
        "tensor_name": ""
      }
    ]
  }
"""
```

### compile model
#### compile the model exported using PTQ/QAT
```python
from hbdk4.compiler import convert, compile, link, March

converted_module = convert(exported_module, March.nash_e)

# gen hbm3
# DEPRECATED: hbm3 will be removed in the future
print("HBM3: compile nash_e mlir")
hbm = compile(
    converted_module, "deploy_hbm3.hbm", March.nash_e, 0, hbdk3_compatible_mode=True
)
if os.path.exists("deploy_hbm3.hbm"):
    print("compile nash_e mlir to deploy_hbm3.hbm successfully")

# gen hbm4
print("HBM4: compile nash_e mlir")

# Method 1: compile into HBO first, then link to HBM
hbo = compile(converted_module, "deploy1_hbm4_.hbo", March.nash_e, 0)
if os.path.exists("deploy1_hbm4_.hbo"):
    print("compile nash_e mlir to deploy1_hbm4_.hbo successfully")
hbm = link([hbo], "deploy1_hbm4_.hbm")
if os.path.exists("deploy1_hbm4_.hbm"):
    print("link hbo to deploy1_hbm4_.hbm successfully")

# Method 2: compile hbm directly
hbm = compile(converted_module, "deploy2_hbm4_.hbm", March.nash_e, 0)
if os.path.exists("deploy2_hbm4_.hbm"):
    print("compile nash_e mlir to deploy2_hbm4_.hbm successfully")
```
### package multiple models into one HBM
```python
from hbdk4.compiler import compile, link, March

conv_hbo_name = "conv_compiled.hbo"
conv_hbo = compile(
    converted_module_1,
    conv_hbo_name,
    march,
    0,
    progress_bar=True,
    advice=0.01,
    balance=2,
)
linear_hbo_name = "linear_compiled.hbo"
linear_hbo = compile(
    converted_module_2,
    linear_hbo_name,
    march,
    0,
    progress_bar=True,
    advice=0.01,
    balance=2,
)
hbm_name = "compiled.hbm"
hbm = link([conv_hbo, linear_hbo], hbm_name)
if os.path.exists(hbm_name):
    print("link two hbos successfully")


# If multiple HBOs have already been generated and you want to package them into one HBM, you can use the following method
from hbdk4.compiler.hbm import Hbo

hbo1 = Hbo("conv_compiled.hbo")
hbo2 = Hbo("linear_compiled.hbo")
hbm = link([hbo1, hbo2], "pack.hbm")
if os.path.exists("pack.hbm"):
    print("link two hbo files successfully")
```

### model static perf
```python
import json
import os

from hbdk4.compiler import hbm_perf

# After successful execution, FPS, latency, and DDR data volume information will be printed, and detailed information can be viewed in the generated HTML file
hbm_perf("deploy.hbm")
if os.path.exists(f"{model_name}.html"):
    print("hbm perf successfully")

# Whether existing json file and check int8 perf info
json_flag = False
ops_flag = False
if os.path.exists(f"{model_name}.json"):
    with open(f"{model_name}.json", "r") as f:
        json_flag = True
        perf_json = json.load(f)
        perf_summary = perf_json["summary"]
        if "BPU OPs per run (effective)" in perf_summary:
            ops_flag = True

assert json_flag, "perf json file not existing"
assert ops_flag, "OPS perf info not existing"
```

### model inference
```python
from hbdk4.compiler import load, Hbm
import numpy as np

hbir = load(bc_path)
hbm = Hbm(hbm_path)

# prepare for random input
inputs = {
    v.name: np.random.rand(*v.type.shape).astype(v.type.np_dtype)
    for v in hbir[0].inputs
}

# hbir and Hbm inference
hbir_outputs = hbir[0].feed(inputs)
hbm_outputs = hbm[0].feed(inputs)

# compare Hbir and hbm outputs
for idx, v in enumerate(hbir[0].outputs):
    hbir_data = hbir_outputs[v.name]
    hbm_data = hbm_outputs[v.name]
    np.testing.assert_equal(
        hbm_data,
        hbir_data,
        "output{} -- {} is not equal".format(idx, v.name),
    )
print("All outputs are equal")
```

### HBIR model modify
#### tensor name modify
```python

from hbdk4.compiler import load

module = load(bc_path)

# get the corresponding FunctionalHelper for the function
func = module.functions[0]

# original func: func @unet_mobilenetv1_cityscapes(tensor<1x3x1024x2048xf32> _input_0) -> tensor<1x1x256x512xsi64> _output_0
print(func)

# modify inputs[0]'s name
func.inputs[0].name = "modified_name_img"
# modify outputs[0]'s name
func.outputs[0].name = "modified_name_output"

# modified func:  func @unet_mobilenetv1_cityscapes(tensor<1x3x1024x2048xf32> modified_name_img) -> tensor<1x1x256x512xsi64> modified_name_output
print(func)
```

#### model desc modify
```python
from hbdk4.compiler import Module

mlir_text = """
module {
  func.func @model(%arg0 : tensor<64x4x3xf32>) -> tensor<64x4x3xf32> {
    %1 = "hbir.abs"(%arg0) : (tensor<64x4x3xf32>) -> tensor<64x4x3xf32>
    return %1 : tensor<64x4x3xf32>
  }
}
"""

module = Module.parse(mlir_text)

# get the corresponding FunctionalHelper for the function
func = module.functions[0]

# modify func's desc
func.desc = "model description"
# modify inputs[0]'s desc
func.inputs[0].desc = "RGB input"
# modify outputs[0]'s desc
func.outputs[0].desc = "gesture"

# model description
print(func.desc)
# RGB input
print(func.inputs[0].desc)
# gesture
print(func.outputs[0].desc)

""" modified model
module {
  func.func @model(%arg0: tensor<64x4x3xf32> {hbir.desc = "RGB input", hbir.id = 1 : i64, hbir.name = "_input_0"}) -> (tensor<64x4x3xf32> {hbir.desc = "gesture", hbir.id = -1 : i64, hbir.name = "_output_0"}) attributes {hbir.desc = "model description"} {
    %0 = "hbir.abs"(%arg0) : (tensor<64x4x3xf32>) -> tensor<64x4x3xf32>
    return %0 : tensor<64x4x3xf32>
  }
}
"""
print(module.module)
```

#### insert nodes
Note: To avoid the new insertion operator not running in some conversion passes, it is recommended to call the insert_xxx api before the convert stage
##### insert pyramid input
```python
from hbdk4.compiler import Module

# nv12
mlir_text = """
module {
  func.func @main(%arg0: tensor<1x32x32x3xf32>, %arg1: tensor<4x1x1x3xf32>) -> tensor<1x32x32x4xf32> {
    %0 = "hbir.conv2d"(%arg0, %arg1) <{dilation = [1, 1], groupNum = 1 : i64, pad = [0, 0, 0, 0], stride = [1, 1]}> : (tensor<1x32x32x3xf32>, tensor<4x1x1x3xf32>) -> tensor<1x32x32x4xf32>
    return %0 : tensor<1x32x32x4xf32>
  }
}
"""
module = Module.parse(mlir_text)
func = module[0]
y, uv = func.inputs[0].insert_image_convert("nv12")
# tensor<1x32x32x1xui8> _input_0_y
print(y)
# tensor<1x16x16x2xui8> _input_0_uv
print(uv)

""" modified model
module {
  func.func @main(%arg0: tensor<1x32x32x1xui8> {hbir.id = 3 : i64, hbir.name = "_input_0_y"}, %arg1: tensor<1x16x16x2xui8> {hbir.id = 4 : i64, hbir.name = "_input_0_uv"}, %arg2: tensor<4x1x1x3xf32> {hbir.id = 2 : i64, hbir.name = "_input_1"}) -> (tensor<1x32x32x4xf32> {hbir.id = -1 : i64, hbir.name = "_output_0"}) {
    %0 = "hbir.image_convert"(%arg0, %arg1) <{mode = #hbdk<ImageConvertMode NV12>}> : (tensor<1x32x32x1xui8>, tensor<1x16x16x2xui8>) -> tensor<1x32x32x3xsi8>
    %1 = "hbir.cast_type"(%0) : (tensor<1x32x32x3xsi8>) -> tensor<1x32x32x3xf32>
    %2 = "hbir.conv2d"(%1, %arg2) <{dilation = [1, 1], groupNum = 1 : i64, pad = [0, 0, 0, 0], stride = [1, 1]}> : (tensor<1x32x32x3xf32>, tensor<4x1x1x3xf32>) -> tensor<1x32x32x4xf32>
    return %2 : tensor<1x32x32x4xf32>
  }
}
"""
print(module.module)

# gray
mlir_text = """
func.func @main(%arg0: tensor<1x32x32x1xf32>, %arg1: tensor<4x1x1x1xf32>) -> tensor<1x32x32x4xf32> {
%0 = "hbir.conv2d"(%arg0, %arg1) : (tensor<1x32x32x1xf32>, tensor<4x1x1x1xf32>) -> tensor<1x32x32x4xf32>
return %0 : tensor<1x32x32x4xf32>
}
"""
module = Module.parse(mlir_text)
func = module[0]
y = func.inputs[0].insert_image_convert("gray")
# tensor<1x32x32x1xui8> _input_0_y
print(y)

""" modified model
module {
  func.func @main(%arg0: tensor<1x32x32x1xui8> {hbir.id = 3 : i64, hbir.name = "_input_0_y"}, %arg1: tensor<4x1x1x1xf32> {hbir.id = 2 : i64, hbir.name = "_input_1"}) -> (tensor<1x32x32x4xf32> {hbir.id = -1 : i64, hbir.name = "_output_0"}) {
    %0 = "hbir.image_convert"(%arg0) <{mode = #hbdk<ImageConvertMode GRAY>}> : (tensor<1x32x32x1xui8>) -> tensor<1x32x32x1xsi8>
    %1 = "hbir.cast_type"(%0) : (tensor<1x32x32x1xsi8>) -> tensor<1x32x32x1xf32>
    %2 = "hbir.conv2d"(%1, %arg1) <{dilation = [1, 1], groupNum = 1 : i64, pad = [0, 0, 0, 0], stride = [1, 1]}> : (tensor<1x32x32x1xf32>, tensor<4x1x1x1xf32>) -> tensor<1x32x32x4xf32>
    return %2 : tensor<1x32x32x4xf32>
  }
}

"""
print(module.module)
```

##### insert image preprocess
```python
from hbdk4.compiler import Module

mlir_text = """
    func.func @main(%arg0: tensor<1x32x32x3xf32>, %arg1: tensor<1x32x32x3xf32>) -> (tensor<1x32x32x4xf32>, tensor<1x32x32x4xf32>) {
    %0 = "hbir.constant"() <{values = dense<3.> : tensor<4x1x1x3xf32>}> : () -> tensor<4x1x1x3xf32>
    %1 = "hbir.conv2d"(%arg0, %0) : (tensor<1x32x32x3xf32>, tensor<4x1x1x3xf32>) -> tensor<1x32x32x4xf32>
    %2 = "hbir.constant"() <{values = dense<4.> : tensor<4x1x1x3xf32>}> : () -> tensor<4x1x1x3xf32>
    %3 = "hbir.conv2d"(%arg1, %2) : (tensor<1x32x32x3xf32>, tensor<4x1x1x3xf32>) -> tensor<1x32x32x4xf32>
    return %1, %3 : tensor<1x32x32x4xf32>, tensor<1x32x32x4xf32>
    }
"""

module = Module.parse(mlir_text)
func = module[0]
func.inputs[0].name = "img1"
func.inputs[1].name = "img2"
func.outputs[0].name = "pred1"
func.outputs[1].name = "pred2"
func.inputs[0].desc = "This is yuv"

node_0 = func.inputs[1].insert_image_preprocess(
    mode=None, divisor=1, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
# tensor<1x32x32x3xsi8> img2
print(node_0)

yuv = func.inputs[0].insert_image_preprocess(
    mode="yuvbt601full2rgb",
    divisor=255,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
# tensor<1x32x32x3xsi8> img1
print(yuv)

""" modified model
module {
  func.func @main(%arg0: tensor<1x32x32x3xsi8> {hbir.desc = "This is yuv", hbir.id = 4 : i64, hbir.name = "img1"}, %arg1: tensor<1x32x32x3xsi8> {hbir.id = 3 : i64, hbir.name = "img2"}) -> (tensor<1x32x32x4xf32> {hbir.id = -1 : i64, hbir.name = "pred1"}, tensor<1x32x32x4xf32> {hbir.id = -2 : i64, hbir.name = "pred2"}) {
    %0 = "hbir.image_preprocess"(%arg0) <{csc = #hbdk<CSCMode YUVBT601FULL2RGB>, divisor = 255 : i64, mean = [4.850000e-01, 4.560000e-01, 4.060000e-01], sd = [2.290000e-01, 2.240000e-01, 2.250000e-01]}> : (tensor<1x32x32x3xsi8>) -> tensor<1x32x32x3xf32>
    %1 = "hbir.image_preprocess"(%arg1) <{csc = #hbdk<CSCMode NONE>, divisor = 1 : i64, mean = [4.850000e-01, 4.560000e-01, 4.060000e-01], sd = [2.290000e-01, 2.240000e-01, 2.250000e-01]}> : (tensor<1x32x32x3xsi8>) -> tensor<1x32x32x3xf32>
    %2 = "hbir.constant"() <{values = dense<3.000000e+00> : tensor<4x1x1x3xf32>}> : () -> tensor<4x1x1x3xf32>
    %3 = "hbir.conv2d"(%0, %2) <{dilation = [1, 1], groupNum = 1 : i64, pad = [0, 0, 0, 0], stride = [1, 1]}> : (tensor<1x32x32x3xf32>, tensor<4x1x1x3xf32>) -> tensor<1x32x32x4xf32>
    %4 = "hbir.constant"() <{values = dense<4.000000e+00> : tensor<4x1x1x3xf32>}> : () -> tensor<4x1x1x3xf32>
    %5 = "hbir.conv2d"(%1, %4) <{dilation = [1, 1], groupNum = 1 : i64, pad = [0, 0, 0, 0], stride = [1, 1]}> : (tensor<1x32x32x3xf32>, tensor<4x1x1x3xf32>) -> tensor<1x32x32x4xf32>
    return %3, %5 : tensor<1x32x32x4xf32>, tensor<1x32x32x4xf32>
  }
}
"""
print(module.module)
```

##### insert rle
```python
from hbdk4.compiler import Module, March, convert

mlir_text = """
  func.func @main(%arg0: tensor<1x32x32x3xf32>, %arg1: tensor<4x1x1x3xf32>) -> (tensor<1x32x32x4xf32>) {
    %0 = "qnt.const_fake_quant"(%arg0) <{bits = 8 : i64, illegal = true, max = [1.270000e+01], min = [-1.280000e+01], narrowRange = false}> : (tensor<1x32x32x3xf32>) -> tensor<1x32x32x3xf32>
    %1 = "qnt.const_fake_quant"(%arg1) <{bits = 8 : i64, illegal = true, max = [1.270000e+02], min = [-1.280000e+02], narrowRange = false}> : (tensor<4x1x1x3xf32>) -> tensor<4x1x1x3xf32>
    %2 = "hbir.conv2d"(%0, %1) <{dilation = [1, 1], groupNum = 1 : i64, pad = [0, 0, 0, 0], stride = [1, 1]}> : (tensor<1x32x32x3xf32>, tensor<4x1x1x3xf32>) -> tensor<1x32x32x4xf32>
    %3 = "qnt.const_fake_quant"(%2) <{bits = 8 : i64, illegal = true, max = [25.761327266693115], min = [-25.96417236328125], narrowRange = false}> : (tensor<1x32x32x4xf32>) -> tensor<1x32x32x4xf32>
    return %3 : tensor<1x32x32x4xf32>
  }
"""

module = Module.parse(mlir_text)
func = module[0]
func.inputs[0].name = "img"
func.inputs[0].desc = "in"
func.inputs[1].name = "weight"
func.outputs[0].name = "pred"
func.outputs[0].desc = "out"

""" modified model
module {
  func.func @main(%arg0: tensor<1x32x32x3xf32> {hbir.desc = "in", hbir.id = 1 : i64, hbir.name = "img"}, %arg1: tensor<4x1x1x3xf32> {hbir.id = 2 : i64, hbir.name = "weight"}) -> (tensor<1x32x32x4xf32> {hbir.desc = "out", hbir.id = -1 : i64, hbir.name = "pred"}) {
    %0 = "qnt.const_fake_quant"(%arg0) <{bits = 8 : i64, illegal = true, max = [1.270000e+01], min = [-1.280000e+01], narrowRange = false}> : (tensor<1x32x32x3xf32>) -> tensor<1x32x32x3xf32>
    %1 = "qnt.const_fake_quant"(%arg1) <{bits = 8 : i64, illegal = true, max = [1.270000e+02], min = [-1.280000e+02], narrowRange = false}> : (tensor<4x1x1x3xf32>) -> tensor<4x1x1x3xf32>
    %2 = "hbir.conv2d"(%0, %1) <{dilation = [1, 1], groupNum = 1 : i64, pad = [0, 0, 0, 0], stride = [1, 1]}> : (tensor<1x32x32x3xf32>, tensor<4x1x1x3xf32>) -> tensor<1x32x32x4xf32>
    %3 = "qnt.const_fake_quant"(%2) <{bits = 8 : i64, illegal = true, max = [25.761327266693115], min = [-25.96417236328125], narrowRange = false}> : (tensor<1x32x32x4xf32>) -> tensor<1x32x32x4xf32>
    return %3 : tensor<1x32x32x4xf32>
  }
}
"""
print(module.module)

converted_module: Module = convert(m=module, march=March.nash_e)
func = converted_module.functions[0]
func.remove_io_op(["Dequantize"])

""" modified model
module attributes {hbdk.perf_stage = 1 : i64, hbdk.target = "B30G"} {
  func.func @main(%arg0: tensor<1x32x32x3xf32> {hbir.desc = "in", hbir.id = 1 : i64, hbir.name = "img"}, %arg1: tensor<4x1x1x3xf32> {hbir.id = 2 : i64, hbir.name = "weight"}) -> (tensor<1x32x32x4xsi8> {hbir.desc = "out", hbir.id = -2 : i64, hbir.name = "pred", hbir.scale = !qnt.uniform<si8:f32, 0.20284509658813477>}) {
    %0 = "hbtl.call"(%arg0) <{diffRank = 0 : i64, isCustom = false, parameters = [[1.000000e-01], [0], false, 0, false], signature = "quant::quantize(Tensor, double[], int64_t[], bool, int64_t, bool) -> (Tensor)"}> : (tensor<1x32x32x3xf32>) -> tensor<1x32x32x3xsi8>
    %1 = "hbtl.call"(%arg1) <{diffRank = 0 : i64, isCustom = false, parameters = [[1.000000e+00], [0], false, 0, false], signature = "quant::quantize(Tensor, double[], int64_t[], bool, int64_t, bool) -> (Tensor)"}> : (tensor<4x1x1x3xf32>) -> tensor<4x1x1x3xsi8>
    %2 = "hbir.constant"() <{values = dense<[[0, 1, 0, 0, 32308, 16], [0, 1, 0, 0, 32308, 16], [0, 1, 0, 0, 32308, 16], [0, 1, 0, 0, 32308, 16]]> : tensor<4x6xsi64>}> : () -> tensor<4x6xsi64>
    %3 = "b30.conv2d"(%0, %1, %2) <{dilation = [1, 1], instanceId = -1 : i64, kernel = [1, 1], operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>, pad = [0, 0, 0, 0], padValue = 0 : i64, processDone = 0 : i32, relu = false, stride = [1, 1]}> ({
    }) : (tensor<1x32x32x3xsi8>, tensor<4x1x1x3xsi8>, tensor<4x6xsi64>) -> tensor<1x32x32x4xsi8>
    return %3 : tensor<1x32x32x4xsi8>
  }
}
"""
print(converted_module.module)

out_node = func.outputs[0].insert_rle()
# tensor<4096x2xui8> pred
print(out_node)

""" modified model
module attributes {hbdk.perf_stage = 1 : i64, hbdk.target = "B30G"} {
  func.func @main(%arg0: tensor<1x32x32x3xf32> {hbir.desc = "in", hbir.id = 1 : i64, hbir.name = "img"}, %arg1: tensor<4x1x1x3xf32> {hbir.id = 2 : i64, hbir.name = "weight"}) -> (tensor<4096x2xui8, {dynamic_dims = [0]}> {hbir.desc = "out", hbir.id = -3 : i64, hbir.name = "pred", hbir.scale = !qnt.uniform<si8:f32, 0.20284509658813477>}) {
    %0 = "hbtl.call"(%arg0) <{diffRank = 0 : i64, isCustom = false, parameters = [[1.000000e-01], [0], false, 0, false], signature = "quant::quantize(Tensor, double[], int64_t[], bool, int64_t, bool) -> (Tensor)"}> : (tensor<1x32x32x3xf32>) -> tensor<1x32x32x3xsi8>
    %1 = "hbtl.call"(%arg1) <{diffRank = 0 : i64, isCustom = false, parameters = [[1.000000e+00], [0], false, 0, false], signature = "quant::quantize(Tensor, double[], int64_t[], bool, int64_t, bool) -> (Tensor)"}> : (tensor<4x1x1x3xf32>) -> tensor<4x1x1x3xsi8>
    %2 = "hbir.constant"() <{values = dense<[[0, 1, 0, 0, 32308, 16], [0, 1, 0, 0, 32308, 16], [0, 1, 0, 0, 32308, 16], [0, 1, 0, 0, 32308, 16]]> : tensor<4x6xsi64>}> : () -> tensor<4x6xsi64>
    %3 = "b30.conv2d"(%0, %1, %2) <{dilation = [1, 1], instanceId = -1 : i64, kernel = [1, 1], operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>, pad = [0, 0, 0, 0], padValue = 0 : i64, processDone = 0 : i32, relu = false, stride = [1, 1]}> ({
    }) : (tensor<1x32x32x3xsi8>, tensor<4x1x1x3xsi8>, tensor<4x6xsi64>) -> tensor<1x32x32x4xsi8>
    %4 = "hbir.reshape"(%3) <{foldable = true, shape = [4096]}> : (tensor<1x32x32x4xsi8>) -> tensor<4096xsi8>
    %5 = "b30.rle"(%4) ({
    }) : (tensor<4096xsi8>) -> tensor<8256xsi8>
    %6 = "hbtl.call"(%5) <{diffRank = 0 : i64, isCustom = false, parameters = [4096, 64], signature = "horizon::RlePostProcess(Tensor, int64_t, int64_t) -> (Tensor)"}> : (tensor<8256xsi8>) -> tensor<4096x2xui8, {dynamic_dims = [0]}>
    return %6 : tensor<4096x2xui8, {dynamic_dims = [0]}>
  }
}
"""
print(converted_module.module)

```

##### insert roi resize
```python
from hbdk4.compiler import Module

mlir_text = """
func.func @main(%arg0: tensor<1x32x32x3xf32>, %arg1: tensor<4x1x1x3xf32>) -> tensor<1x32x32x4xf32> {
  %0 = "hbir.conv2d"(%arg0, %arg1) : (tensor<1x32x32x3xf32>, tensor<4x1x1x3xf32>) -> tensor<1x32x32x4xf32>
  return %0 : tensor<1x32x32x4xf32>
}
"""

module = Module.parse(mlir_text)
func = module[0]
func.inputs[0].name = "img"
func.inputs[0].desc = "test"
func.inputs[1].name = "weight"
func.outputs[0].name = "pred"

y, uv, roi = func.inputs[0].insert_roi_resize("nv12")

# tensor<1x8192x8192x1xui8> img_y
print(y)
# tensor<1x4096x4096x2xui8> img_uv
print(uv)
# tensor<1x4xsi32> img_roi
print(roi)

""" modified model
module {
  func.func @main(%arg0: tensor<1x8192x8192x1xui8, {dynamic_dims = [-2, -3]}> {hbir.desc = "test", hbir.id = 3 : i64, hbir.name = "img_y"}, %arg1: tensor<1x4096x4096x2xui8, {dynamic_dims = [-2, -3]}> {hbir.desc = "test", hbir.id = 4 : i64, hbir.name = "img_uv"}, %arg2: tensor<1x4xsi32> {hbir.desc = "test", hbir.id = 5 : i64, hbir.name = "img_roi"}, %arg3: tensor<4x1x1x3xf32> {hbir.id = 2 : i64, hbir.name = "weight"}) -> (tensor<1x32x32x4xf32> {hbir.id = -1 : i64, hbir.name = "pred"}) {
    %0 = "hbir.roi_resize"(%arg0, %arg1, %arg2) <{expansionMode = #hbdk<ExpansionMode constant>, interpolateMode = #hbdk<InterpolationMode bilinear>, mode = #hbdk<ImageConvertMode NV12>, padValue = [0, -128], size = [32, 32]}> : (tensor<1x8192x8192x1xui8, {dynamic_dims = [-2, -3]}>, tensor<1x4096x4096x2xui8, {dynamic_dims = [-2, -3]}>, tensor<1x4xsi32>) -> tensor<1x32x32x3xsi8>
    %1 = "hbir.cast_type"(%0) : (tensor<1x32x32x3xsi8>) -> tensor<1x32x32x3xf32>
    %2 = "hbir.conv2d"(%1, %arg3) <{dilation = [1, 1], groupNum = 1 : i64, pad = [0, 0, 0, 0], stride = [1, 1]}> : (tensor<1x32x32x3xf32>, tensor<4x1x1x3xf32>) -> tensor<1x32x32x4xf32>
    return %2 : tensor<1x32x32x4xf32>
  }
}
"""
print(module.module)
```

##### insert transpose
```python
from hbdk4.compiler import Module

mlir_text = """
func.func @main(%arg0: tensor<1x32x32x3xf32>, %arg1: tensor<4x1x1x3xf32>) -> tensor<1x32x32x4xf32> {
  %0 = "hbir.conv2d"(%arg0, %arg1) : (tensor<1x32x32x3xf32>, tensor<4x1x1x3xf32>) -> tensor<1x32x32x4xf32>
  return %0 : tensor<1x32x32x4xf32>
}
"""

module = Module.parse(mlir_text)
func = module[0]
func.inputs[0].name = "img"
func.inputs[0].desc = "in"
func.inputs[1].name = "weight"
func.outputs[0].name = "pred"
func.outputs[0].desc = "out"

""" modified model
module {
  func.func @main(%arg0: tensor<1x32x32x3xf32> {hbir.desc = "in", hbir.id = 1 : i64, hbir.name = "img"}, %arg1: tensor<4x1x1x3xf32> {hbir.id = 2 : i64, hbir.name = "weight"}) -> (tensor<1x32x32x4xf32> {hbir.desc = "out", hbir.id = -1 : i64, hbir.name = "pred"}) {
    %0 = "hbir.conv2d"(%arg0, %arg1) <{dilation = [1, 1], groupNum = 1 : i64, pad = [0, 0, 0, 0], stride = [1, 1]}> : (tensor<1x32x32x3xf32>, tensor<4x1x1x3xf32>) -> tensor<1x32x32x4xf32>
    return %0 : tensor<1x32x32x4xf32>
  }
}
"""
print(module.module)

in_node = func.inputs[0].insert_transpose([2, 3, 0, 1])
# tensor<32x3x1x32xf32> img
print(in_node)

out_node = func.outputs[0].insert_transpose([2, 3, 1, 0])
# tensor<32x4x32x1xf32> pred
print(out_node)

""" modified model
module {
  func.func @main(%arg0: tensor<32x3x1x32xf32> {hbir.desc = "in", hbir.id = 3 : i64, hbir.name = "img"}, %arg1: tensor<4x1x1x3xf32> {hbir.id = 2 : i64, hbir.name = "weight"}) -> (tensor<32x4x32x1xf32> {hbir.desc = "out", hbir.id = -2 : i64, hbir.name = "pred"}) {
    %0 = "hbir.transpose"(%arg0) <{dims = [2, 3, 0, 1]}> : (tensor<32x3x1x32xf32>) -> tensor<1x32x32x3xf32>
    %1 = "hbir.conv2d"(%0, %arg1) <{dilation = [1, 1], groupNum = 1 : i64, pad = [0, 0, 0, 0], stride = [1, 1]}> : (tensor<1x32x32x3xf32>, tensor<4x1x1x3xf32>) -> tensor<1x32x32x4xf32>
    %2 = "hbir.transpose"(%1) <{dims = [2, 3, 1, 0]}> : (tensor<1x32x32x4xf32>) -> tensor<32x4x32x1xf32>
    return %2 : tensor<32x4x32x1xf32>
  }
}
"""
print(module.module)

```

##### insert split
```python
from hbdk4.compiler import Module

mlir_text = """
func.func @main(%arg0: tensor<1x32x32x3xf32>, %arg1: tensor<4x1x1x3xf32>) -> tensor<1x32x32x4xf32> {
  %0 = "hbir.conv2d"(%arg0, %arg1) : (tensor<1x32x32x3xf32>, tensor<4x1x1x3xf32>) -> tensor<1x32x32x4xf32>
  return %0 : tensor<1x32x32x4xf32>
}
"""

module = Module.parse(mlir_text)
func = module[0]
func.inputs[0].name = "img"
func.inputs[0].desc = "its img"
func.inputs[1].name = "weight"
func.inputs[1].desc = "its weight"
func.outputs[0].name = "pred"
func.outputs[0].desc = "its out"

# func @main(tensor<1x32x32x3xf32> img, tensor<4x1x1x3xf32> weight) -> tensor<1x32x32x4xf32> pred
print(module)


# Split the dimension 0 of the input parameter into multiple input parameters
res_list = func.inputs[1].insert_split(0)

# tensor<1x1x1x3xf32> weight_0
print(res_list[0])

""" modified model
module {
  func.func @main(%arg0: tensor<1x32x32x3xf32> {hbir.desc = "its img", hbir.id = 1 : i64, hbir.name = "img"}, %arg1: tensor<1x1x1x3xf32> {hbir.desc = "its weight", hbir.id = 3 : i64, hbir.name = "weight_0"}, %arg2: tensor<1x1x1x3xf32> {hbir.desc = "its weight", hbir.id = 4 : i64, hbir.name = "weight_1"}, %arg3: tensor<1x1x1x3xf32> {hbir.desc = "its weight", hbir.id = 5 : i64, hbir.name = "weight_2"}, %arg4: tensor<1x1x1x3xf32> {hbir.desc = "its weight", hbir.id = 6 : i64, hbir.name = "weight_3"}) -> (tensor<1x32x32x4xf32> {hbir.desc = "its out", hbir.id = -1 : i64, hbir.name = "pred"}) {
    %0 = "hbir.concat"(%arg1, %arg2, %arg3, %arg4) <{dim = 0 : i64}> : (tensor<1x1x1x3xf32>, tensor<1x1x1x3xf32>, tensor<1x1x1x3xf32>, tensor<1x1x1x3xf32>) -> tensor<4x1x1x3xf32>
    %1 = "hbir.conv2d"(%arg0, %0) <{dilation = [1, 1], groupNum = 1 : i64, pad = [0, 0, 0, 0], stride = [1, 1]}> : (tensor<1x32x32x3xf32>, tensor<4x1x1x3xf32>) -> tensor<1x32x32x4xf32>
    return %1 : tensor<1x32x32x4xf32>
  }
}
"""
print(module.module)

res_list_1 = func.outputs[0].insert_split(3)
# tensor<1x32x32x1xf32> pred_0
print(res_list_1[0])

""" modified model
module {
  func.func @main(%arg0: tensor<1x32x32x3xf32> {hbir.desc = "its img", hbir.id = 1 : i64, hbir.name = "img"}, %arg1: tensor<1x1x1x3xf32> {hbir.desc = "its weight", hbir.id = 3 : i64, hbir.name = "weight_0"}, %arg2: tensor<1x1x1x3xf32> {hbir.desc = "its weight", hbir.id = 4 : i64, hbir.name = "weight_1"}, %arg3: tensor<1x1x1x3xf32> {hbir.desc = "its weight", hbir.id = 5 : i64, hbir.name = "weight_2"}, %arg4: tensor<1x1x1x3xf32> {hbir.desc = "its weight", hbir.id = 6 : i64, hbir.name = "weight_3"}) -> (tensor<1x32x32x1xf32> {hbir.desc = "its out", hbir.id = -2 : i64, hbir.name = "pred_0"}, tensor<1x32x32x1xf32> {hbir.desc = "its out", hbir.id = -3 : i64, hbir.name = "pred_1"}, tensor<1x32x32x1xf32> {hbir.desc = "its out", hbir.id = -4 : i64, hbir.name = "pred_2"}, tensor<1x32x32x1xf32> {hbir.desc = "its out", hbir.id = -5 : i64, hbir.name = "pred_3"}) {
    %0 = "hbir.concat"(%arg1, %arg2, %arg3, %arg4) <{dim = 0 : i64}> : (tensor<1x1x1x3xf32>, tensor<1x1x1x3xf32>, tensor<1x1x1x3xf32>, tensor<1x1x1x3xf32>) -> tensor<4x1x1x3xf32>
    %1 = "hbir.conv2d"(%arg0, %0) <{dilation = [1, 1], groupNum = 1 : i64, pad = [0, 0, 0, 0], stride = [1, 1]}> : (tensor<1x32x32x3xf32>, tensor<4x1x1x3xf32>) -> tensor<1x32x32x4xf32>
    %2 = "hbir.slice"(%1) <{begin = [0, 0, 0, 0], end = [1, 32, 32, 1], foldable = false, step = [1, 1, 1, 1]}> : (tensor<1x32x32x4xf32>) -> tensor<1x32x32x1xf32>
    %3 = "hbir.slice"(%1) <{begin = [0, 0, 0, 1], end = [1, 32, 32, 2], foldable = false, step = [1, 1, 1, 1]}> : (tensor<1x32x32x4xf32>) -> tensor<1x32x32x1xf32>
    %4 = "hbir.slice"(%1) <{begin = [0, 0, 0, 2], end = [1, 32, 32, 3], foldable = false, step = [1, 1, 1, 1]}> : (tensor<1x32x32x4xf32>) -> tensor<1x32x32x1xf32>
    %5 = "hbir.slice"(%1) <{begin = [0, 0, 0, 3], end = [1, 32, 32, 4], foldable = false, step = [1, 1, 1, 1]}> : (tensor<1x32x32x4xf32>) -> tensor<1x32x32x1xf32>
    return %2, %3, %4, %5 : tensor<1x32x32x1xf32>, tensor<1x32x32x1xf32>, tensor<1x32x32x1xf32>, tensor<1x32x32x1xf32>
  }
}
"""
print(module.module)
```

#### remove adjacent nodes in the input and output of the model
```python
from hbdk4.compiler import Module

mlir_text = """
module {
  func.func @model(%arg0 : tensor<64x4x3xf32>, %arg1 : tensor<64x4x3xf32>) -> tensor<64x4x3xf32> {
    %0 = "hbir.add"(%arg0, %arg1) : (tensor<64x4x3xf32>, tensor<64x4x3xf32>) -> tensor<64x4x3xf32>
    %1 = "hbir.mul"(%arg0, %0) : (tensor<64x4x3xf32>, tensor<64x4x3xf32>) -> tensor<64x4x3xf32>

    %2 = "hbir.abs"(%1) : (tensor<64x4x3xf32>) -> tensor<64x4x3xf32>
    return %2 : tensor<64x4x3xf32>
  }
}
"""

module = Module.parse(mlir_text)
removable, reason = module[0].inputs[0].is_removable
# False
print(removable)
# '_input_0 has multiple uses. cannot remove.'
print(reason)

removable, reason = module[0].inputs[1].is_removable
# False
print(removable)
# '%0 = "hbir.add"(%arg0, %arg1) : (tensor<64x4x3xf32>, tensor<64x4x3xf32>) -> tensor<64x4x3xf32> loc("-":4:10) is not a single input and single output op'
print(reason)

removable, reason = module[0].outputs[0].is_removable
# True
print(removable)
# ''
print(reason)

res = module[0].outputs[0].remove_attached_op()
# (True, '')
print(res)
""" modified model
module {
  func.func @model(%arg0: tensor<64x4x3xf32> {hbir.id = 1 : i64, hbir.name = "_input_0"}, %arg1: tensor<64x4x3xf32> {hbir.id = 2 : i64, hbir.name = "_input_1"}) -> (tensor<64x4x3xf32> {hbir.id = -2 : i64, hbir.name = "_output_0"}) {
    %0 = "hbir.add"(%arg0, %arg1) : (tensor<64x4x3xf32>, tensor<64x4x3xf32>) -> tensor<64x4x3xf32>
    %1 = "hbir.mul"(%arg0, %0) : (tensor<64x4x3xf32>, tensor<64x4x3xf32>) -> tensor<64x4x3xf32>
    return %1 : tensor<64x4x3xf32>
  }
}
"""
print(module.module)
```

### HBM modify
```python
import json
from hbdk4.compiler import hbm_extract_desc, hbm_update_desc

hbm4_model_name = "hbm4_model.hbm"

# step1: get hbm desc dict
hbm4_desc_dict = hbm_extract_desc(hbm4_model_name)
""" original hbm desc
{
  "march": "NashE",
  "models": {
    "hbm4_model": {
      "desc": null,
      "desc_type": null,
      "input_features": {
        "input_0": {
          "desc": null,
          "desc_type": null
        },
        "input_1": {
          "desc": null,
          "desc_type": null
        }
      },
      "output_features": {
        "output_0": {
          "desc": null,
          "desc_type": null
        }
      }
    }
  }
}
"""
print(json.dumps(hbm4_desc_dict, indent=2))

# step2: modify hbm desc dict
# adjust name and description information
# graph desc/name modification
hbm4_desc_dict["models"]["hbm4_model"]["desc"] = "set_new_graph_desc"
hbm4_desc_dict["models"]["hbm4_model"]["desc_type"] = "string"
hbm4_desc_dict["models"]["hbm4_model"]["new_name"] = "set_new_graph_name"

# input desc/name modification
print(hbm4_desc_dict)
hbm4_desc_dict["models"]["hbm4_model"]["input_features"]["input_0"][
    "desc"
] = "set_new_input_desc"
hbm4_desc_dict["models"]["hbm4_model"]["input_features"]["input_0"][
    "desc_type"
] = "string"
hbm4_desc_dict["models"]["hbm4_model"]["input_features"]["input_0"][
    "new_name"
] = "set_new_input_name"

# output desc/name modification
hbm4_desc_dict["models"]["hbm4_model"]["output_features"]["output_0"][
    "desc"
] = "set_new_output_desc"
hbm4_desc_dict["models"]["hbm4_model"]["output_features"]["output_0"][
    "desc_type"
] = "string"
hbm4_desc_dict["models"]["hbm4_model"]["output_features"]["output_0"][
    "new_name"
] = "set_new_output_name"

# step3: update model file
hbm_update_desc(hbm4_model_name, hbm4_desc_dict)

new_hbm4_desc_dict = hbm_extract_desc(hbm4_model_name)

""" new hbm desc
{
  "march": "NashE",
  "models": {
    "set_new_graph_name": {
      "desc": "set_new_graph_desc",
      "desc_type": "string",
      "input_features": {
        "input_1": {
          "desc": null,
          "desc_type": null
        },
        "set_new_input_name": {
          "desc": "set_new_input_desc",
          "desc_type": "string"
        }
      },
      "output_features": {
        "set_new_output_name": {
          "desc": "set_new_output_desc",
          "desc_type": "string"
        }
      }
    }
  }
}
"""
print(json.dumps(new_hbm4_desc_dict, indent=2))
```
