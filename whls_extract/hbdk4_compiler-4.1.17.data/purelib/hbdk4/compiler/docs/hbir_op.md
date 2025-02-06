# HBIR Operator Definition

## `hbir.abs` (::mlir::hbdk::hbir::AbsOp)

HBIR tensor abs.

Return tensor after the operation, which has the same shape as the input.
Traits: CommonVerifier, EltwiseLike, MoveF16CastLike, SameOperandsAndResultShape

Interfaces: DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.acos` (::mlir::hbdk::hbir::AcosOp)

HBIR tensor acos.

Return tensor after the operation, which has the same shape as the input.
Traits: CommonVerifier, LutLike, SameOperandsAndResultShape

Interfaces: DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.acosh` (::mlir::hbdk::hbir::AcoshOp)

HBIR tensor acosh.

Return tensor after the operation, which has the same shape as the input.
Traits: CommonVerifier, LutLike, SameOperandsAndResultShape

Interfaces: DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.add` (::mlir::hbdk::hbir::AddOp)

HBIR tensor addition.

Applies addition operator element-wise, $y_i=lhs_i+rhs_i$.

------

Note:
* Our arithmetic operator support broadcast.

------

Prototype: Pytorch add.

Traits: Broadcastable, CommonVerifier, EltwiseLike

Interfaces: CalibOp, DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `lhs` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type
| `rhs` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type


## `hbir.asin` (::mlir::hbdk::hbir::AsinOp)

HBIR tensor asin.

Return tensor after the operation, which has the same shape as the input.
Traits: CommonVerifier, LutLike, SameOperandsAndResultShape

Interfaces: DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.asinh` (::mlir::hbdk::hbir::AsinhOp)

HBIR tensor asinh.

Return tensor after the operation, which has the same shape as the input.
Traits: CommonVerifier, LutLike, SameOperandsAndResultShape

Interfaces: DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.atan` (::mlir::hbdk::hbir::AtanOp)

HBIR tensor atan.

Return tensor after the operation, which has the same shape as the input.
Traits: CommonVerifier, LutLike, SameOperandsAndResultShape

Interfaces: DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.atanh` (::mlir::hbdk::hbir::AtanhOp)

HBIR tensor atanh.

Return tensor after the operation, which has the same shape as the input.
Traits: CommonVerifier, LutLike, SameOperandsAndResultShape

Interfaces: DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.avg_pool` (::mlir::hbdk::hbir::AvgPoolOp)

HBIR n-D average pooling(only support 1d and 2d currently).

Applies a nD average pooling over input.

In the simplest case, the output value of the operator with input size $(N, H, W, C)$,
output $(N, H_{out}, W_{out}, C)$ and kernel size $(ker_{h}, ker_{w})$ can be precisely described as:

$ out(N_i, h, w, C_j) = \frac{1} { ker_h *ker_w } \sum_{m = 0} ^ { ker_h - 1 }\sum_{n = 0} ^
{ ker_w - 1 } input(N_i, stride[0]\times h + m, stride[1] \times w + n, C_j) $

where $h,w$ respectively represent the size of H and W.

------

Note:

* parameters has the same manner as the Conv2D operator, the same goes for the output size.

* ceilMode controls output's compute is mode of floor or ceil, it's default value is false.

------

Shape:

* Input: $(N, H_{in}, W_{in}, C)$ or $(H_{in}, W_{in}, C)$ or $(*, H_{in}, W_{in})$

* Output: $(N, H_{out}, W_{out}, C)$ or $(H_{out}, W_{out}, C)$ or $(*, H_{out}, W_{out}, C)$,
where $*$ represents any number of dimension.

$ H_{out} = \lfloor {\frac{H_{in} + padding[0] + padding[2] - kernel[0]} {stride[0]} + 1}\rfloor $

$ W_{out} = \lfloor {\frac{W_{in} + padding[1] + padding[3] - kernel[1]} {stride[1]} + 1}\rfloor $

if ceilMode = true, please use ceil replace floor in the above output formula.

------

Prototype: Pytorch avg_pool.

Traits: CommonVerifier, PoolLike, StencilLike

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `kernel` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `stride` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `pad` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `dilation` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `ceilMode` | ::mlir::BoolAttr | bool attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or  values or none type


## `hbir.batchnorm` (::mlir::hbdk::hbir::BatchNormOp)

Hbir Batch Normalize

Applies Batch Normalization over each dimension of input. This compute can be precisely described as:

$ y = \frac{x-mean[x]}{\sqrt{Var[x]+\epsilon}}*weight+bias $

This mean and standard-deviation are calculated per-dimension over the batches
and weight and bias are learnable parameter vectors of the input size.

------

Note:
* eps - a value added to the denominator for numerical stability.
* $mean(x)$ and $Var[x]$'s shape are $(C)$.
* weight and bias are learnable scalar.

------

Shape:
* Input: $(N,H,W,C)$ or $(N,M,H,W,C)$ or $(H,W,C)$ or $(*,H,W,C)$, where $*$ reprensent any number of dimension.
* Output: same shape as input.

------

Prototype: Pytorch BatchNorm.

Traits: CommonVerifier, Misc, SameVariadicOperandSize

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, RoiInfer, SchedInterface, SchedTemp, ShapeInference, Tiling

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `eps` | ::mlir::FloatAttr | 64-bit float attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type
| `mean` | 1D tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type
| `var` | 1D tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type
| `weight` | 1D tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type
| `bias` | 1D tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type


## `hbir.bev_pool_v2` (::mlir::hbdk::hbir::BevPoolV2Op)

HBIR bev_pool_v2 op, cpu operator, from mmdet3d, no corresponding operator in torch/onnx

Convert several planar image inputs into bev image outputs, thus providing support for data processing under bird's eye view.

------

Parameters:
* depth: the depth tensor.
* feat: the feat tensor.
* ranks_depth: Stores the index value of depth.
* ranks_feat: Stores the index value of feat.
* ranks_bev: Stores the Voxel index value of the valid bev space.
* interval_starts: Each element marks the starting point of each "continuation segment" of the ranks_bev feat.
* interval_lengths: Each element identifies the length of each "continuous segment" of the ranks_bev feat.
* bev_feat_shape: output's shape. Aligned with the public version of cudu kernel, no permute(0, 4, 1, 2, 3) operation is performed in the kernel. And can support rank>=4.

------

Shape:
* depth: (B, N, D, fH, fW)
* feat: (B, N, fH, fW, C)
* ranks_depth: (N_points, )
* ranks_feat: (N_points, )
* ranks_bev: (N_points, )
* interval_starts: (N_pillar, )
* interval_lengths: (N_pillar, )
* output: shape same as bev_feat_shape, (B, D_Z, D_Y, D_X, C)


Traits: CommonVerifier, Misc

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `bev_feat_shape` | ::mlir::ArrayAttr | 64-bit integer array attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `depth` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type
| `feat` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type
| `ranks_depth` | tensor of 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer values or none type
| `ranks_feat` | tensor of 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer values or none type
| `ranks_bev` | tensor of 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer values or none type
| `interval_starts` | tensor of 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer values or none type
| `interval_lengths` | tensor of 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.bitshift` (::mlir::hbdk::hbir::BitShiftOp)

HBIR bitshift op

logic shift, positive value as right shift.
Traits: CommonVerifier, EltwiseLike

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 32-bit signed integer or 16-bit signed integer or 8-bit signed integer values
| `rshift` | 1D tensor of 8-bit signed integer values

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 32-bit signed integer or 16-bit signed integer or 8-bit signed integer values


## `hbir.bitwise_and` (::mlir::hbdk::hbir::BitwiseAndOp)

HBIR tensor and.

Applies 'bitwise and' operator element-wise, $y_i = lhs_i \& rhs_i$.
Traits: Broadcastable, CommonVerifier, EltwiseLike, SameOperandsAndResultElementType

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `lhs` | tensor of 8-bit signed integer or 16-bit signed integer values or none type
| `rhs` | tensor of 8-bit signed integer or 16-bit signed integer values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit signed integer or 16-bit signed integer values or none type


## `hbir.bitwise_not` (::mlir::hbdk::hbir::BitwiseNotOp)

HBIR tensor bitwise not.


Traits: CommonVerifier, EltwiseLike, SameOperandsAndResultElementType

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 8-bit signed integer or 16-bit signed integer values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit signed integer or 16-bit signed integer values or none type


## `hbir.bitwise_or` (::mlir::hbdk::hbir::BitwiseOrOp)

HBIR tensor or.

Applies 'bitwise or' operator element-wise, $y_i = lhs_i | rhs_i$.
Traits: Broadcastable, CommonVerifier, EltwiseLike, SameOperandsAndResultElementType

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `lhs` | tensor of 8-bit signed integer or 16-bit signed integer values or none type
| `rhs` | tensor of 8-bit signed integer or 16-bit signed integer values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit signed integer or 16-bit signed integer values or none type


## `hbir.bitwise_xor` (::mlir::hbdk::hbir::BitwiseXorOp)

HBIR tensor xor.

Applies 'bitwise xor' operator element - wise, $y_i = lhs_i xor rhs_i$.
Traits: Broadcastable, CommonVerifier, EltwiseLike, SameOperandsAndResultElementType

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `lhs` | tensor of 8-bit signed integer or 16-bit signed integer values or none type
| `rhs` | tensor of 8-bit signed integer or 16-bit signed integer values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit signed integer or 16-bit signed integer values or none type


## `hbir.cast_type` (::mlir::hbdk::hbir::CastTypeOp)

elemental type cast operation

Data are actually moved.
Traits: CommonVerifier, Misc, NaiveRoiInfer, NaiveTiling, SameOperandsAndResultShape

Interfaces: CastOpInterface, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), NonBatchAxesInfer, Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `forceSaturate` | ::mlir::BoolAttr | bool attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values


## `hbir.ceil` (::mlir::hbdk::hbir::CeilOp)

HBIR tensor ceil.

Return tensor after the operation, which has the same shape as the input.
Traits: CommonVerifier, EltwiseLike, SameOperandsAndResultShape

Interfaces: DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.clip` (::mlir::hbdk::hbir::ClipOp)

HBIR Clip Op.

Clamps all elements in input into the range $[min, max] $.Assume min_value and max_value be min and max, respectively, this performs:

$ y_i = min(max(x_i, min\_value_i), max\_value_i) $

------

Note:
* min(min_value): lower-bound of the range to be clamped to
* max(max_value): upper-bound of the range to be clamped to

------

Shape:
* Input: (*), where * means any number of dimensions.
* Output: (*), same shape as the input.

------

Prototype: Pytorch hardtanh.

Traits: CommonVerifier, LutLike, MoveF16CastLike, NaiveRoiInfer, NaiveTiling, SameOperandsAndResultShape

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `min` | ::mlir::Attribute | 64-bit float attribute or 64-bit signless integer attribute
| `max` | ::mlir::Attribute | 64-bit float attribute or 64-bit signless integer attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.concat` (::mlir::hbdk::hbir::ConcatOp)

Concatenates tensors along one dimension

Concatenates the given sequence of seq tensors in the given dimension. No elemental type conversion.

------

Note:
* dim - the dimension over which the tensors are concatenated.

------

Prototype: Pytorch cat.

Traits: CommonVerifier, MoveF16CastLike, MoveLike, NaiveTiling, SameElementType

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, Layout, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), NonBatchAxesInfer, Perf, PortAccess, Quantizable, RoiInfer, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `dim` | ::mlir::IntegerAttr | 64-bit signless integer attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `inputs` | variadic of tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type


## `hbir.constant` (::mlir::hbdk::hbir::ConstantOp)

HBIR constant generation op.

Generate a constant with specified type and value
Traits: CommonVerifier, Constant, NoFuseFp16TypeLike

Interfaces: HBTLExecutable, HbdkExecutorInterface, NoMemoryEffect (MemoryEffectOpInterface), NonBatchAxesInfer, Perf, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `values` | ::mlir::ElementsAttr | constant vector/tensor attribute

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values


## `hbir.conv2d` (::mlir::hbdk::hbir::Conv2dOp)

HBIR 2-D convolution.

Applies a 2D convolution over an input signal composed of several input channels.

In the simplest case,
the output value of the layer with input size $(N, H_{in}, W_{in}, C_{in})$
and output $(N, H_{out}, W_{out}, C_{out})$ can be precisely descibed as:

$ out(N_i,C_{out_j}) = bias(C_{out_j}) + \sum_{k=0}^{C_{in} - 1}weight(C_{out_j},k) \star input(N_i,k) $

where $\star$ is the valid 2D [cross-correlation](https://www.mathworks.com/help/signal/ref/xcorr2.html) operation,
$N$ is the batch size, $C$ denotes a number of channels, $H$ and $W$ are the size of pixels.

------

Note:

* stride controls the stride for the cross-correlation, an integer array with 2 elements, default value is (1,1).

* padding controls the amount of padding applied to the input, an integer array with 4 elements, the padding sequences is (h_begin,w_begin,h_end,w_end), default value is (0,0,0,0).

* dilation controls the spacing between kernel points, an integer array with 2 elements, default value is (0,0). It's harder to describe, but this [link](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md) has a nice visualization of what dilation does.

* groups controls the connections between inputs and outputs, an integer variable, default value is 1.

* Weight:  $(C_{out}, KH, KW, C_{in})$, bias shape = $C_{out}$ where KW and KH represent kernel's height and width, respectively.

------

Shape:

* Input: $(N,H_{in},W_{in},C_{in})$ or $(H_{in},W_{in},C_{in})$ or $(N,M,H_{in},W_{in},C_{in})$ or $(*,H_{in},W_{in},C_{in})$,
where * represent any number of dimension.

* Output: $(N,H_{out},W_{out},C_{out})$ or $(H_{out},W{out},C_{out})$ or $(N,M,H_{out},W_{out},C_{out})$ or $(*,H_{out},W_{out},C_{out})$

$ H_{out}=\lfloor \frac{H_{in} + padding[0] + padding[2] - dilation[0]\times(kernel[0]-1)-1}{stride[0]}+1\rfloor $
$ W_{out}=\lfloor \frac{W_{in}+padding[1]+padding[3]-dilation[1]\times(kernel[1]-1)-1}{stride[1]}+1\rfloor $

------

Prototype: Pytorch convolution.

Traits: CommonVerifier, ConvLike, NoFuseFp16TypeLike, StencilLike

Interfaces: CalibOp, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `stride` | ::mlir::ArrayAttr | 64-bit integer array attribute with exactly 2 elements
| `pad` | ::mlir::ArrayAttr | 64-bit integer array attribute with exactly 4 elements
| `dilation` | ::mlir::ArrayAttr | 64-bit integer array attribute with exactly 2 elements
| `groupNum` | ::mlir::IntegerAttr | 64-bit signless integer attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or  values or none type
| `weight` | 4D tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or  values or none type
| `bias` | 1D tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or  values or none type


## `hbir.conv3d` (::mlir::hbdk::hbir::Conv3dOp)

HBIR 3-D convolution.

Applies a 3D convolution over an input signal composed of several input planes.

In the simplest case, the output value of the layer with input size $(N, C_{in}, D, H, W)$
and output $(N, C_{out}, D_{out}, H_{out}, W_{out})$ can be precisely described as:

$ out(N_i, C_{out_j}) = bias(C_{out_j}) +
\sum_{k = 0}^{C_{in} - 1} weight(C_{out_j}, k) \star input(N_i, k) $

where $\star$ is the valid 3D [cross-correlation] operation,
$N$ is the batch size, $C$ denotes a number of channels, $D$, $H$ and $W$ are the size of pixels.

------

Note:

* stride controls the stride for the cross-correlation, an integer array with 3 elements, default value is (1,1,1).

* padding controls the amount of padding applied to the input,  an integer array with 5 elements,
the padding sequences is (d_begin,h_begin,w_begin,d_end,h_end,w_end), default value is (0,0,0,0,0,0).

* dilation controls the spacing between kernel points, an integer array with 3 elements, default value is (0,0,0).
It's harder to describe, but this [link](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)
has a nice visualization of what dilation does.

* groups controls the connections between inputs and outputs, an integer variable, default value is 1.

* Weight: $(C_{out}, KD, KH, KW, C_{in})$, bias shape = $C_{out}$ where KW and KH represent kernel's height and width, respectively.

------

Shape:

* Input: $(N,D_{in},H_{in},W_{in},C_{in})$ or $(D_{in},H_{in},W_{in},C_{in})$ or $(*,D_{in},H_{in},W_{in},C_{in})$,
where * represent any number of dimension.

* Output: $(N,D_{out},H_{out},W_{out},C_{out})$ or $(D_{out},H_{out},W{out},C_{out})$ or $(*,D_{out},H_{out},W_{out},C_{out})$.

$ D_{out}=\lfloor \frac{D_{in} + padding[0] + padding[3] - dilation[0]\times(kernel[0]-1)-1}{stride[0]}+1\rfloor $

$ H_{out}=\lfloor \frac{H_{in} + padding[1] + padding[4] - dilation[1]\times(kernel[1]-1)-1}{stride[1]}+1\rfloor $

$ W_{out}=\lfloor \frac{W_{in} + padding[2] + padding[5] - dilation[2]\times(kernel[2]-1)-1}{stride[2]}+1\rfloor $

------

Prototype: Pytorch convolution.

Traits: CommonVerifier, ConvLike, StencilLike

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `stride` | ::mlir::ArrayAttr | 64-bit integer array attribute with exactly 3 elements
| `pad` | ::mlir::ArrayAttr | 64-bit integer array attribute with exactly 6 elements
| `dilation` | ::mlir::ArrayAttr | 64-bit integer array attribute with exactly 3 elements
| `groupNum` | ::mlir::IntegerAttr | 64-bit signless integer attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or  values or none type
| `weight` | 5D tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or  values or none type
| `bias` | 1D tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or  values or none type


## `hbir.conv` (::mlir::hbdk::hbir::ConvOp)

HBIR convolution.

Applies a convolution over an input signal composed of several input planes.

rank in convolution is an integer greater than or equal to 1.

In the simplest case, for 2D convolution, rank=2
the output value of the layer with input size $(N, H_{in}, W_{in}, C_{in})$
and output $(N, H_{out}, W_{out}, C_{out})$ can be precisely descibed as:

$ out(N_i,C_{out_j}) = bias(C_{out_j}) + \sum_{k=0}^{C_{in} - 1}weight(C_{out_j},k) \star input(N_i,k) $

where $\star$ is the valid 2D [cross-correlation](https://www.mathworks.com/help/signal/ref/xcorr2.html) operation,
$N$ is the batch size, $C$ denotes a number of channels, $H$ and $W$ are the size of pixels.

------

Note:

* stride controls the stride for the cross-correlation, an integer array with rank elements, default value is (1,1,...).

* padding controls the amount of padding applied to the input, an integer array with 2*rank elements, when rank=2, the padding sequences is (h_begin,w_begin,h_end,w_end), default value is (0,0,0,0), when n=3, the padding sequences is (d_begin,h_begin,w_begin,d_end,h_end,w_end), default value is (0,0,0,0,0,0).

* dilation controls the spacing between kernel points, an integer array with rank elements, default value is (0,0,...). It's harder to describe, but this [link](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md) has a nice visualization of what dilation does.

* groups controls the connections between inputs and outputs, an integer variable, default value is 1.

* Weight for 2D:  $(C_{out}, KH, KW, C_{in})$, bias shape = $C_{out}$ where KW and KH represent kernel's height and width, respectively.

------

Shape for 2D:

* Input: $(N,H_{in},W_{in},C_{in})$ or $(H_{in},W_{in},C_{in})$ or $(N,M,H_{in},W_{in},C_{in})$ or $(*,H_{in},W_{in},C_{in})$,
where * represent any number of dimension.

* Output: $(N,H_{out},W_{out},C_{out})$ or $(H_{out},W{out},C_{out})$ or $(N,M,H_{out},W_{out},C_{out})$ or $(*,H_{out},W_{out},C_{out})$

$ H_{out}=\lfloor \frac{H_{in} + padding[0] + padding[2] - dilation[0]\times(kernel[0]-1)-1}{stride[0]}+1\rfloor $
$ W_{out}=\lfloor \frac{W_{in}+padding[1]+padding[3]-dilation[1]\times(kernel[1]-1)-1}{stride[1]}+1\rfloor $

------

Prototype: Pytorch convolution.

Traits: CommonVerifier, ConvLike, NoFuseFp16TypeLike, StencilLike

Interfaces: DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `stride` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `pad` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `dilation` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `groupNum` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `channelLast` | ::mlir::BoolAttr | bool attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or  values or none type
| `weight` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or  values or none type
| `bias` | 1D tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or  values or none type


## `hbir.convtranspose` (::mlir::hbdk::hbir::ConvTransposeOp)

HBIR transposed conv1d or conv2d or conv3d op.

Inverse operation of Conv op.

------

Parameters:
* input: the input tensor
* weight: the deconvolution kernel
* stride: same as conv1d's stride, [s_w];same as conv2d's stride, [s_h, s_w];same as conv3d's stride, [s_d, s_h, s_w]
* pad: 1d pad information for clipping output [w_left,w_right];2d pad information for clipping output [h_top,w_left,h_bottom,w_right];3d pad information for clipping output [d_front,h_top,w_left,d_back,h_bottom,w_right]
* dilation: same as conv1d's dilation, [d_w];same as conv2d's dilation, [d_h, d_w];same as conv3d's dilation, [d_d, d_h, d_w]
* group: same as conv1d's or conv2d's or conv3d's group

------

1d Shape:
* input: $(*, w, in\_channel)$
* weight: $(in\_channel, kw, out\_channel / group)$
* output: $(*, wo, out\_channel)$
* bias: $out\_channel$

where:

$ wo = (w - 1) * stride[0] - (pad[0] + pad[1]) + dilation[0] * (kw - 1) + 1 $

2d Shape:
* input: $(*, h, w, in\_channel)$
* weight: $(in\_channel, kh, kw, out\_channel / group)$
* output: $(*, ho, wo, out\_channel)$
* bias: $out\_channel$

where:

$ ho = (h - 1) * stride[0] - (pad[0] + pad[2]) + dilation[0] * (kh - 1) + 1 $
$ wo = (w - 1) * stride[1] - (pad[1] + pad[3]) + dilation[1] * (kw - 1) + 1 $

3d Shape:
* input: $(*, d, h, w, in\_channel)$
* weight: $(in\_channel, kd, kh, kw, out\_channel / group)$
* output: $(*, do, ho, wo, out\_channel)$
* bias: $out\_channel$

where:

$ do = (d - 1) * stride[0] - (pad[0] + pad[3]) + dilation[0] * (kd - 1) + 1 $
$ ho = (h - 1) * stride[1] - (pad[1] + pad[4]) + dilation[1] * (kh - 1) + 1 $
$ wo = (w - 1) * stride[2] - (pad[2] + pad[5]) + dilation[2] * (kw - 1) + 1 $

Traits: CommonVerifier, ConvLike, StencilLike

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `stride` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `pad` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `dilation` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `groupNum` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `illegalWeight` | ::mlir::BoolAttr | bool attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or  values or none type
| `weight` | 3D/4D/5D tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or  values or none type
| `bias` | 1D tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type values

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or  values or none type


## `hbir.cos` (::mlir::hbdk::hbir::CosOp)

HBIR tensor cos.

Return tensor after the operation, which has the same shape as the input.
Traits: CommonVerifier, LutLike, SameOperandsAndResultShape

Interfaces: DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.cosh` (::mlir::hbdk::hbir::CoshOp)

HBIR tensor cosh.

Return tensor after the operation, which has the same shape as the input.
Traits: CommonVerifier, LutLike, SameOperandsAndResultShape

Interfaces: DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.cumsum` (::mlir::hbdk::hbir::CumSumOp)

HBIR cumsum.

Performs cumulative sum of the input elements along the given axis.
By default, it will do the sum inclusively meaning the first element is copied as is.
Through an exclusive attribute, this behavior can change to exclude the first element.
It can also perform summation in the opposite direction of the axis. For that, set reverse attribute to 1.

  Args:
        input (Tensor): the input tensor.
        output (Tensor): Output tensor of the same type as input with cumulative sums of the input elements
  Attribute:
        axis (int): Must be in the range [-rank(input), rank(input)-1]. Negative value means counting dimensions from the back.
        exclusive (int): Must be 0 or 1, defaut is 0. 0 means the first element is copied to output, 1 will not.
        reverse (int): Must be 0 or 1, defaut is 0. 1 means performing summation in the opposite direction of the axis.

------

Prototype: Pytorch cumsum.

Traits: CommonVerifier, Misc, SameOperandsAndResultShape

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `axis` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `exclusive` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `reverse` | ::mlir::IntegerAttr | 64-bit signless integer attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type


## `hbir.custom` (::mlir::hbdk::hbir::CustomOp)

HBIR custom op

CPU custom operator placeholder.


------

Parameters:
* inputs - tensor inputs, which type is list
* Other non-tensor arguments for CPU operators

Returns:
* outputs - list of tensor

Traits: CommonVerifier, LeapOp, Misc

Interfaces: HBTLExecutable, HbdkExecutorInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `srcPath` | ::mlir::StringAttr | string attribute
| `entryFuncName` | ::mlir::StringAttr | string attribute
| `includeDirs` | ::mlir::StringAttr | string attribute
| `libraryDirs` | ::mlir::StringAttr | string attribute
| `extraCompileArgs` | ::mlir::StringAttr | string attribute
| `extraLinkArgs` | ::mlir::StringAttr | string attribute
| `runtimeLibraryDirs` | ::mlir::StringAttr | string attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `inputs` | variadic of tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values

### Results:

| Result | Description |
| :----: | ----------- |
| `outputs` | variadic of tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values


## `hbir.deform_conv2d` (::mlir::hbdk::hbir::DeformConv2dOp)

HBIR deformable 2-D convolution.

Applies a deformable 2D convolution over an input signal composed of several input channels.

------

Note:

* offset controls the offset for the sampling locations in the convolution kernel.

* mask controls different weights to different positions in the convolution kernel.

* stride controls the stride for the cross-correlation, an integer array with 2 elements, default value is (1,1).

* padding controls the amount of padding applied to the input, an integer array with 4 elements, the padding sequences is (h_begin,w_begin,h_end,w_end), default value is (0,0,0,0).

* dilation controls the spacing between kernel points, an integer array with 2 elements, default value is (0,0). It's harder to describe, but this [link](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md) has a nice visualization of what dilation does.

* groups controls the connections between inputs and outputs, an integer variable, default value is 1.

* offsetGroup controls the connections between inputs and offset, an integer variable, default value is 1.

* weight:  $(C_{out}, KH, KW, C_{in})$, bias shape = $C_{out}$ where KW and KH represent kernel's height and width, respectively.

------

Shape:

* Input: $(N,H_{in},W_{in},C_{in})$ or $(*,H_{in},W_{in},C_{in})$, where * represent any number of dimension.

* Offset: $(N,H_{out},W_{out},2\timesoffset_groups\timeskernel[0]\timeskernel[1])$ or $(*,H_{out},W_{out},2\timesoffset_groups\timeskernel[0]\timeskernel[1])$, where * represent any number of dimension.

* Mask: $(N,H_{out},W_{out},offset_groups\timeskernel[0]\timeskernel[1])$ or $(*,H_{out},W_{out},offset_groups\timeskernel[0]\timeskernel[1])$, where * represent any number of dimension.

* Output: $(N,H_{out},W_{out},C_{out})$ or $(H_{out},W{out},C_{out})$ or $(*,H_{out},W_{out},C_{out})$

$ H_{out}=\lfloor \frac{H_{in}+padding[0]+padding[2]-dilation[0]\times(kernel[0]-1)-1}{stride[0]}+1\rfloor $
$ W_{out}=\lfloor \frac{W_{in}+padding[1]+padding[3]-dilation[1]\times(kernel[1]-1)-1}{stride[1]}+1\rfloor $

------

Prototype: Pytorch convolution.

Traits: CommonVerifier, Misc, NoFuseFp16TypeLike, SameVariadicOperandSize, StencilLike

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `stride` | ::mlir::ArrayAttr | 64-bit integer array attribute with exactly 2 elements
| `pad` | ::mlir::ArrayAttr | 64-bit integer array attribute with exactly 4 elements
| `dilation` | ::mlir::ArrayAttr | 64-bit integer array attribute with exactly 2 elements
| `groupNum` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `offsetGroupNum` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `useMask` | ::mlir::BoolAttr | bool attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or  values or none type
| `weight` | 4D tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or  values or none type
| `offset` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or  values or none type
| `mask` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or  values or none type
| `bias` | 1D tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or  values or none type


## `hbir.div` (::mlir::hbdk::hbir::DivOp)

HBIR tensor division.

Applies division operator element-wise, $y_i=lhs_i\div rhs_i$.

------

Note:
* Our arithmetic operator support broadcast.

------

Prototype: Pytorch div.

Traits: Broadcastable, CommonVerifier, EltwiseLike

Interfaces: DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `lhs` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type
| `rhs` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type


## `hbir.dpp` (::mlir::hbdk::hbir::DppOp)

HBIR detection post process.

Legacy Operator: Detection Process.

------

Note

  * The input features only supports shift quantization.
  * On J5 and later platforms, DPP will be split into Filter OP and CPU operators. Filter Op will run on BPU.

------

Shape:

  * Input:
    * [1 - 5] input features. Normally, the input feature size is 1/2, 1/4, ..., or 1/32 of the original input image.
    * anchors: anchor table of dpp op.
    * anchorNum: anchor number of each input branch.
    * filterThreshold: threshold for score value.
    * nmsThreshold: threshold for nms operation.
    * nmsMargin: score margin of nms operation.
    * seed: seed for random() function.
    * useClipping: clipping flag for each input branch. if set, the box which exceeds the original image will be clipped.
    * stride: input features' stride.
    * clsOffset: class offset of each input branch.
    * imageSize: the image size, if clippling is set, all box should within this range.
    * inputShift: the input feature's quantization coefficient. for legacy mode, only shift quantization is supported.
    * legacyMode: legacy mode to be compatible with J2J3.
    * useBpuFilter: use Filter OP on BPU to accelerate dpp.
    * maxBoxNum: max box num for output
  * Output:
    * boxData: The output box data with format [coord0, coord1, coord2, coord3, max_score, class_index, pad, pad] @ int16.

Traits: CommonVerifier, Misc

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, Layout, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `anchors` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `anchorNum` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `filterThreshold` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `nmsThreshold` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `nmsMargin` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `seed` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `useClipping` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `stride` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `clsOffset` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `imageSize` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `inputShift` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `legacyMode` | ::mlir::BoolAttr | bool attribute
| `useBpuFilter` | ::mlir::BoolAttr | bool attribute
| `maxBoxNum` | ::mlir::IntegerAttr | 64-bit signless integer attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `inputs` | variadic of tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `boxData` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.dynamic_slice` (::mlir::hbdk::hbir::DynamicSliceOp)

Slice a tensor out of a tensor, support dynamic input, begin, end, step, axes. Same as onnx slice.

Slice uses the starts, ends, axes and steps inputs to select a sub-tensor of its input data tensor.

------

Note:
* starts -1-D tensor of starting indices of corresponding axis in axes.
* ends -1-D tensor of ending indices (exclusive) of corresponding axis in axes.
* axes -1-D tensor of axes that starts and ends apply to.
* steps -1-D tensor of slice step of corresponding axis in axes.


------

Prototype: onnx slice.

Traits: CommonVerifier, Misc, SameVariadicOperandSize

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type
| `starts` | 1D tensor of 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer values or none type
| `ends` | 1D tensor of 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer values or none type
| `axes` | 1D tensor of 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer values or none type
| `steps` | 1D tensor of 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type


## `hbir.equal` (::mlir::hbdk::hbir::EqualOp)

HBIR tensor equal.

Determines whether two tensors are equal element by element - wise, $y_i = (lhs_i == rhs_i) $.
Traits: Broadcastable, CommonVerifier, EltwiseLike

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `lhs` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type
| `rhs` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of  values or none type


## `hbir.erf` (::mlir::hbdk::hbir::ErfOp)

HBIR tensor erf.

Return tensor after the operation, which has the same shape as the input.
Traits: CommonVerifier, LutLike, SameOperandsAndResultShape

Interfaces: DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.exp` (::mlir::hbdk::hbir::ExpOp)

HBIR tensor exp.

Applies exponential operator element - wise, $y=e^{x}$.

------

Note:
* Returns a new tensor with the exponential of the elements of the input tensor input.


Traits: CommonVerifier, LutLike, SameOperandsAndResultShape

Interfaces: DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.fake_cast` (::mlir::hbdk::hbir::FakeCastOp)

fake elemental type cast operation

Cast float input to specified dtype, and then cast back to the same float type.
Traits: CommonVerifier, Misc, SameElementType, SameOperandsAndResultShape

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `dtype` | ::mlir::TypeAttr | any type attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 32-bit float or 16-bit float values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 32-bit float or 16-bit float values


## `hbir.filter` (::mlir::hbdk::hbir::FilterOp)

HBIR filter op

Filter data by comparing its correspond score and given threshold.
Filter constains two steps: Apply reduce max on channel c of input to get filter score and index, followed by filter of input supported by hardware.

Traits: CommonVerifier, Misc

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `channelBegin` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `channelEnd` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `threshold` | ::mlir::FloatAttr | 64-bit float attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values

### Results:

| Result | Description |
| :----: | ----------- |
| `maxValue` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type
| `maxIndex` | tensor of 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer values or none type
| `filterCoord` | tensor of 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer values or none type
| `filterData` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type


## `hbir.flip` (::mlir::hbdk::hbir::FlipOp)

HBIR flip

Reverse the order of the input tensor along given axis in dims.

------

Parametes:
* input: the input tensor.
* dims: axis need to reverse.

------

Shape:
* input: $(*)$, where * represents any dimension.
* output: same as the input.

Traits: CommonVerifier, MoveF16CastLike, MoveLike

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `dims` | ::mlir::ArrayAttr | 64-bit integer array attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type


## `hbir.floor` (::mlir::hbdk::hbir::FloorOp)

HBIR tensor floor.

Return tensor after the operation, which has the same shape as the input.
Traits: CommonVerifier, EltwiseLike, SameOperandsAndResultShape

Interfaces: DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.fpp` (::mlir::hbdk::hbir::FppOp)

HBIR Filter Post Process.

Operator: Filter Detection Process.

------

Note

  * The input features only supports shift quantization.

------

Shape:

  * Input:
    * [1 - 5] input features from bpu filter raw data.
    * anchors: anchor table to calculate boxes.
    * anchorNum: anchor number of each input branch.
    * filterThreshold: threshold for score value.
    * nmsThreshold: threshold for nms operation.
    * nmsMargin: score margin of nms operation.
    * seed: seed for random() function.
    * headerPadSize: padding size of filtered box header, bayes is 0, nash is 8.
    * dataChannel: the origin filter input data channel
    * useClipping: clipping flag for each input branch. if set, the box which exceeds the original image will be clipped.
    * stride: input features' stride.
    * clsOffset: class offset of each input branch.
    * imageSize: the image size, if clippling is set, all box should within this range.
    * inputShift: the input feature's quantization coefficient. for legacy mode, only shift quantization is supported.
    * legacyMode: legacy mode to be compatible with J2J3.
    * maxBoxNum: config max box num for output, if not 4096, output is static shape.
    * dataRank: the origin filter input data rank, since bpu filter is rank1 output.
  * Output:
    * boxData: The output box data with format [coord0, coord1, coord2, coord3, max_score, class_index, pad, pad] @ int16.

Traits: CommonVerifier, Misc

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, Layout, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `anchors` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `anchorNum` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `filterThreshold` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `nmsThreshold` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `nmsMargin` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `seed` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `headerPadSize` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `dataChannel` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `useClipping` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `stride` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `clsOffset` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `imageSize` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `inputShift` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `legacyMode` | ::mlir::BoolAttr | bool attribute
| `maxBoxNum` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `dataRank` | ::mlir::IntegerAttr | 64-bit signless integer attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `inputs` | variadic of tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `boxData` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.gelu` (::mlir::hbdk::hbir::GELUOp)

HBIR GELU activation.

Applies gelu funciton element-wise. Gelu functon defined as:

$ Gelu(x) = xP(X\leq x) = x*\phi(x) $

------

Note:
* Gelu is shorthand for Gaussian Error Linear Unit.
* approximate - support 'none' or 'tanh' mode

------

Shape:
* Input: (*), where * means any number of dimensions.
* Output: (*), same shape as the input.

------

Prototype: Pytorch gelu.

Traits: CommonVerifier, LutLike, NaiveRoiInfer, NaiveTiling, SameOperandsAndResultShape

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `approximate` | ::mlir::hbdk::GeluApproximateModeAttr | gelu approximate mode

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.gather_elements` (::mlir::hbdk::hbir::GatherElementsOp)

HBIR gather op for onnx GatherElements.

HBIR gather op for onnx GatherElements.
Traits: CommonVerifier, Misc, MoveF16CastLike

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, IndexOpInterface, NoMemoryEffect (MemoryEffectOpInterface), NonBatchAxesInfer, Perf, Quantizable, RandomAccessInterface, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `dim` | ::mlir::IntegerAttr | 64-bit signless integer attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type
| `indices` | tensor of 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type


## `hbir.gather_nd` (::mlir::hbdk::hbir::GatherNdOp)

HBIR gather_nd op for onnx gather_nd.

HBIR gather_nd op for onnx gather_nd.
Traits: CommonVerifier, Misc

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, IndexOpInterface, NoMemoryEffect (MemoryEffectOpInterface), NonBatchAxesInfer, Perf, Quantizable, RandomAccessInterface, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `batchDim` | ::mlir::IntegerAttr | 64-bit signless integer attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type
| `indices` | tensor of 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values


## `hbir.greater_equal` (::mlir::hbdk::hbir::GreaterEqualOp)

HBIR tensor greater_equal.

Applies greater_equal operator element - wise, $y_i = lhs_i > = rhs_i$.
Traits: Broadcastable, CommonVerifier, EltwiseLike

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `lhs` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type
| `rhs` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of  values or none type


## `hbir.greater` (::mlir::hbdk::hbir::GreaterOp)

HBIR tensor greater.

Applies greater operator element - wise, $y_i = lhs_i > rhs_i$.
Traits: Broadcastable, CommonVerifier, EltwiseLike

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `lhs` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type
| `rhs` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of  values or none type


## `hbir.grid_sample` (::mlir::hbdk::hbir::GridSampleOp)

HBIR grid_sample.

From the input and a flow-field grid, computes the output using input values and pixel locations from the grid.

------

Shape:
* input: input of shape $(*, H_{in}, W_{in}, C_{in}) $, where * represent any number of dimension.
* grid: flow - field of shape $(*, H_{out}, W_{out}, 2)$
* output: $(*, H_{out}, W_{out}, C_{in})$

Traits: CommonVerifier, Expansion, Round, SampleLike

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, RandomAccessInterface, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `mode` | ::mlir::hbdk::InterpolationModeAttr | interpolation mode for all march
| `expansionMode` | ::mlir::hbdk::ExpansionModeAttr | mode to expand input feature on H/W
| `alignCorner` | ::mlir::BoolAttr | bool attribute
| `padValue` | ::mlir::Attribute | 64-bit float attribute or 64-bit signless integer attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type
| `grid` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.identity` (::mlir::hbdk::hbir::IdentityOp)

Identity operator.

Return tensor copy from input tensor.
Traits: CommonVerifier, MoveLike

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type


## `hbir.index` (::mlir::hbdk::hbir::IndexOp)

HBIR index op for aten::index_select

Returns a new tensor which indexes the :attr:`input` tensor along dimension
:attr:`dim` using the entries in :attr:`index` which is a `LongTensor`.

The returned tensor has the same number of dimensions as the original tensor
(:attr:`input`).  The :attr:`dim`\ th dimension has the same size as the length
of :attr:`index`; other dimensions have the same size as in the original tensor.

Args:
      input (Tensor): the input tensor.
      dim (int): the dimension in which we index
      index (IntTensor or LongTensor): the tensor containing the indices to index

Traits: CommonVerifier, Misc, MoveF16CastLike

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, IndexOpInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, RandomAccessInterface, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `dim` | ::mlir::IntegerAttr | 64-bit signless integer attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type
| `index` | tensor of 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type


## `hbir.layernorm` (::mlir::hbdk::hbir::LayerNormOp)

Hbir Layer Normalize

Applies Layer Normalization over a mini - batch of inputs. This compute can be precisely described as:

$ y = \frac{x - mean[x]} {\sqrt{Var[x] +\epsilon}} * weight + bias $

The Mean and standard-deviation are calculated over the last D dimensions, where D is the dimension of normalized_shape(dims).

------

Note:
* Unlike Batch Normalization, which compute mean and standard - deviation, LayerNormalization compute these in single sample's different dimension.
* dims controls normalized_shape.

------

Shape:
* Input: $(N, *)$
* Output: $(N, *)$ (same shape as input)

------

Prototype: Pytorch layerNorm.


Traits: CommonVerifier, Misc, SameVariadicOperandSize

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, Layout, NoMemoryEffect (MemoryEffectOpInterface), NonBatchAxesInfer, Perf, Quantizable, RoiInfer, SchedInterface, SchedTemp, ShapeInference, Tiling

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `dims` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `eps` | ::mlir::FloatAttr | 64-bit float attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type
| `weight` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type
| `bias` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.leaky_relu` (::mlir::hbdk::hbir::LeakyReLUOp)

HBIR Leaky ReLU Op.

Applies LeakyRelu funciton element - wise.LeakyRelu function defined as:

$ LeakyRelu(x) = max(0, x) + slope * min(0, x) $

------

Note:
* slope - Controls the angle of the negative slope.

------

Shape:
* Input: (*), where * means any number of dimensions.
* Output: (*), same shape as the input.

------

Prototype: Pytorch leaky_relu.

Traits: CommonVerifier, LutLike, NaiveRoiInfer, NaiveTiling, SameElementType, SameOperandsAndResultShape

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `slop` | ::mlir::FloatAttr | 64-bit float attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.less_equal` (::mlir::hbdk::hbir::LessEqualOp)

HBIR tensor less_equal.

Applies less_equal operator element - wise, $y_i = lhs_i < = rhs_i$.
Traits: Broadcastable, CommonVerifier, EltwiseLike

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `lhs` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type
| `rhs` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of  values or none type


## `hbir.less` (::mlir::hbdk::hbir::LessOp)

HBIR tensor less.

Applies less operator element - wise, $y_i = lhs_i < rhs_i$.
Traits: Broadcastable, CommonVerifier, EltwiseLike

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `lhs` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type
| `rhs` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of  values or none type


## `hbir.linear` (::mlir::hbdk::hbir::LinearOp)

HBIR Linear.

Applies a linear transformation to the incoming data: $y = xW ^ T + b$

------

Note:
* Weight's shape is $(C_{in},C_{out})$.

------

Shape:
* Input: $(*, C_{in})$ where $*$ represents any number of dimensions and $C_{in}$ = in_features.
* Output: $(*, C_{out})$ where all but the last dimension are the same shape as the input and $C_{out}$ = out_features.

------

Prototype: pytorch linear.

Traits: CommonVerifier, LinearLike

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or  values or none type
| `weight` | 2D tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or  values or none type
| `bias` | 1D tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or  values


## `hbir.log` (::mlir::hbdk::hbir::LogOp)

HBIR tensor log.

Applies natural logarithm operator element - wise, $y = \log_e{(x)}$.

------

Note:
* Returns a new tensor with the natural logarithm of the elements of input.


Traits: CommonVerifier, LutLike, SameOperandsAndResultShape

Interfaces: DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.log_softmax` (::mlir::hbdk::hbir::LogSoftmaxOp)

HBIR LogSoftmax Op.

Applies the LogSoftmax function to an n - dimensional input Tensor rescaling them so that the elements of the n-dimensional output.
The output tensor has the same dimension and shape as the input with values in the range [-inf, 0)

LogSoftmax function is defined as:

$ LogSoftmax(x_i) = log(\frac{exp(x_i)} {\sum_jexp(x_j)}) $

------

Shape:
* Input(*), where * means, any number of additional dimensions.
* Output(*), same shape as the input.

------

Prototype: Pytorch LogSoftmax.

Traits: CommonVerifier, LutLike, NaiveRoiInfer, NaiveTiling, SameOperandsAndResultShape

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `dim` | ::mlir::IntegerAttr | 64-bit signless integer attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.logical_and` (::mlir::hbdk::hbir::LogicalAndOp)

HBIR tensor and.

Applies 'logical and' operator element - wise, $y_i = lhs_i \&\& rhs_i$.
Traits: Broadcastable, CommonVerifier, EltwiseLike, SameOperandsElementType

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `lhs` | tensor of 8-bit signed integer or 16-bit signed integer or  or 16-bit float or  values or none type
| `rhs` | tensor of 8-bit signed integer or 16-bit signed integer or  or 16-bit float or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of  values or none type


## `hbir.logical_not` (::mlir::hbdk::hbir::LogicalNotOp)

HBIR tensor logical not, output bool.


Traits: CommonVerifier, EltwiseLike, SameOperandsAndResultShape, SameOperandsElementType

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 8-bit signed integer or 16-bit signed integer or 16-bit float or  or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of  values or none type


## `hbir.logical_or` (::mlir::hbdk::hbir::LogicalOrOp)

HBIR tensor or.

Applies 'logical or' operator element - wise, $y_i = lhs_i || rhs_i$.
Traits: Broadcastable, CommonVerifier, EltwiseLike, SameOperandsElementType

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `lhs` | tensor of 8-bit signed integer or 16-bit signed integer or  or 16-bit float or  values or none type
| `rhs` | tensor of 8-bit signed integer or 16-bit signed integer or  or 16-bit float or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of  values or none type


## `hbir.logical_xor` (::mlir::hbdk::hbir::LogicalXorOp)

HBIR tensor xor.

Applies 'logical xor' operator element - wise, $y_i = lhs_i xor rhs_i$.
Traits: Broadcastable, CommonVerifier, EltwiseLike, SameOperandsElementType

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `lhs` | tensor of 8-bit signed integer or 16-bit signed integer or  or 16-bit float or  values or none type
| `rhs` | tensor of 8-bit signed integer or 16-bit signed integer or  or 16-bit float or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of  values or none type


## `hbir.masked_select` (::mlir::hbdk::hbir::MaskedSelectOp)

HBIR masked_select op

Returns a new 1-D tensor which indexes the input tensor according to the boolean mask mask which is a BoolTensor.

The shapes of the mask tensor and the input tensor don't need to match, but they must be broadcastable.

Traits: Broadcastable, CommonVerifier, EltwiseLike

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type
| `mask` | tensor of  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type


## `hbir.matmul` (::mlir::hbdk::hbir::MatMulOp)

HBIR Matrix Multiplication.

Applies matrix multiplication between two inputs: $C = A \times B$,
where $\times$ means matrix multiplication.

------

Note:
* If both tensors are 1-dimensional, the dot product (scalar) is returned
* If both arguments are 2-dimensional, the matrix-matrix product is returned
* If the first argument is 1-dimensional and the second argument is 2-dimensional, the matrix-vector product is returned
* If the first argument is 2-dimensional and the second argument is 1-dimensional, the matrix-vector product is returned
* If both arguments are at least 1-dimensional and at least one argument is N-dimensional (where N > 2), then a batched matrix multiply is returned

------

Shape:
* lhs: $(B_{l}, M, C)$, where $B_{l}$ represent any number of dimension.
* rhs: $(B_{r}, C, N)$, where $B_{r}$ represent any number of dimension.
* output: $(B_{o}, M, N)$, where $B_{o}$ represent represent the result of broadcast between $B_{l}$ and $V_{r}$.

------

Prototype: Pytorch Matmul.

Traits: CommonVerifier, MatmulLike

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `lhs` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or  values or none type
| `rhs` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or  values


## `hbir.max` (::mlir::hbdk::hbir::MaxOp)

HBIR tensor max.

Applies maximum operator element-wise, $y_i=max(lhs_i,rhs_i)$.

------

Note:
* Our arithmetic operator support broadcast.

Traits: Broadcastable, CommonVerifier, EltwiseLike, MoveF16CastLike, SameElementType

Interfaces: DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `lhs` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type
| `rhs` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type


## `hbir.max_pool` (::mlir::hbdk::hbir::MaxPoolOp)

HBIR n-D max pooling(supports 1d, 2d and 3d).

Applies a n-D max Pooling over an input.

In the 2d case, for example, the output value of the operator with input size $(N, H, W, C)$,
output $(N, H_{out}, W_{out}, C)$ and kernel size $(ker_{h}, ker_{w})$ can be precisely described as:

$ out(N_i, h, w, C_j) = \frac{1} { ker_h *ker_w }\max_{m = 0} ^ { ker_h - 1 }\max_{n = 0} ^
{ ker_w - 1 } input(N_i, stride[0] \times h + m, stride[1] \times w + n, C_j) $

where $h,w$ respectively represent the size of H and W.

------

Note:

* parameters has the same manner as the Conv2D operator, the same goes for the output size.
* ceilMode controls output 's compute is mode of floor or ceil, it' s default value is false.

------

Shape:

* Input: $(N, H_{in}, W_{in}, C)$ or $(H_{in}, W_{in}, C)$ or $(*, H_{in}, W_{in})$

* Output: $(N, H_{out}, W_{out}, C)$ or $(H_{out}, W_{out}, C)$ or $(*, H_{out}, W_{out}, C)$,
where $*$ represents any number of dimension.

$ H_{out} = \lfloor {\frac{H_{in} + padding[0] + padding[2] - kernel[0]} {stride[0]} + 1}\rfloor $

$ W_{out} = \lfloor {\frac{W_{in} + padding[1] + padding[3] - kernel[1]} {stride[1]} + 1}\rfloor $

if ceilMode = true, please use ceil replace floor in the above ouput formula.

------

Prototype: Pytorch max_pool.

Traits: CommonVerifier, MoveF16CastLike, PoolLike, StencilLike

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `kernel` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `stride` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `pad` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `dilation` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `ceilMode` | ::mlir::BoolAttr | bool attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or  values or none type


## `hbir.min` (::mlir::hbdk::hbir::MinOp)

HBIR tensor min.

Applies minimum operator element-wise, $y_i=min(lhs_i,rhs_i)$.

------

Note:
* Our arithmetic operator support broadcast.

Traits: Broadcastable, CommonVerifier, EltwiseLike, MoveF16CastLike, SameElementType

Interfaces: DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `lhs` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type
| `rhs` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type


## `hbir.mod` (::mlir::hbdk::hbir::ModOp)

Hbir get modulo of two dividing tensors

Computes the modulo function.
It is equivalent to the operator x1 % x2

------

Parameters:
* sameSignAsDividend: result has the same sign as dividend or divisor. Default true means same sign as dividend.

Traits: Broadcastable, CommonVerifier, EltwiseLike

Interfaces: DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `sameSignAsDividend` | ::mlir::BoolAttr | bool attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `lhs` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type
| `rhs` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type


## `hbir.mul` (::mlir::hbdk::hbir::MulOp)

HBIR tensor multiplication.

Applies multiplication operator element-wise, $y_i=lhs_i\times rhs_i$.

------

Note:
* Our arithmetic operator support broadcast.

------

Prototype: Pytorch mul.

Traits: Broadcastable, CommonVerifier, EltwiseLike

Interfaces: CalibOp, DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `lhs` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type
| `rhs` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type


## `hbir.nan_to_num` (::mlir::hbdk::hbir::NanToNumOp)

HBIR tensor nan_to_num.

Replaces NaN, positive infinity, and negative infinity values in input with the values specified by nan, posinf, and neginf.

------

Note:
* Returns a new tensor with the replace value of the elements of input.


Traits: CommonVerifier, EltwiseLike, SameOperandsAndResultShape

Interfaces: DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `nan` | ::mlir::FloatAttr | 64-bit float attribute
| `posinf` | ::mlir::FloatAttr | 64-bit float attribute
| `neginf` | ::mlir::FloatAttr | 64-bit float attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type values or none type


## `hbir.neg` (::mlir::hbdk::hbir::NegOp)

HBIR tensor neg.

Applies negation operator element - wise, $y = x\times - 1$.

------

Note:
* Returns a new tensor with the negative of the elements of input.


Traits: CommonVerifier, EltwiseLike, MoveF16CastLike, SameOperandsAndResultShape

Interfaces: DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.nms` (::mlir::hbdk::hbir::NonMaxSuppressionOp)

HBIR NonMaxSuppression

Filter out boxes that have high intersection-over-union (IOU) overlap with previously selected boxes.

------

Parameters:
* boxes: The input boxes.
* scores: The dimension to sort along.
* boxType: The format of the box data. Support mode are "xyxy": [x1,y1,x2,y2],  "yxyx": [y1,x1,y2,x2], "xywh": [x_center, y_center, width, height]
* iouThreshold: Representing the threshold for deciding whether boxes overlap too much with respect to IOU. It is scalar. Value range [0, 1]. Default to 0.
* scoreThreshold: Representing the threshold for deciding when to remove boxes based on score. It is a scalar.
* maxOutputBoxesPerClass: Representing the maximum number of boxes to be selected per batch per class. It is a scalar. Default to 1.

------

Shape:
* boxes: $(batch, boxesNum, 4)$
* scores: $(batch, classNum, boxesNum)$
* Indices: $(N, 3)$. N is euqal batch*classNum*min(boxesNum,maxOutputBoxesPerClass). The last dim means selected index, format is [batch_index, class_index, box_index].


------

Prototype: ONNX NonMaxSuppression

Traits: CommonVerifier, Misc

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `mode` | ::mlir::hbdk::BoxTypeModeAttr | boxes type mode mode for all march
| `iouThreshold` | ::mlir::FloatAttr | 64-bit float attribute
| `scoreThreshold` | ::mlir::FloatAttr | 64-bit float attribute
| `maxOutputBoxesPerClass` | ::mlir::IntegerAttr | 64-bit signless integer attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `boxes` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type
| `scores` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `indices` | tensor of 64-bit signed integer values or none type


## `hbir.nonzero` (::mlir::hbdk::hbir::NonZeroOp)

HBIR nonzero op for torch.nonzero

Find all indices in tensor that are not 0

Traits: CommonVerifier, Misc

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer values or none type


## `hbir.pad` (::mlir::hbdk::hbir::PadOp)

Pad at both edges of Tensor.

Padding at the begin and end position with constant / border value.
Traits: CommonVerifier, Expansion, Foldable, MoveF16CastLike, MoveLike, SameElementType

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, Layout, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), NonBatchAxesInfer, Perf, Quantizable, RoiInfer, SchedInterface, SchedTemp, ShapeInference, Tiling

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `begin` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `end` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `expansionMode` | ::mlir::hbdk::ExpansionModeAttr | mode to expand input feature on H/W
| `padValue` | ::mlir::Attribute | 64-bit float attribute or 64-bit signless integer attribute
| `foldable` | ::mlir::BoolAttr | bool attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type


## `hbir.point_pillar_preprocess` (::mlir::hbdk::hbir::PointPillarPreProcessOp)

HBIR point pillar preprocess op.

HBIR point pillar preprocess.Voxelization and Normalization
Traits: CommonVerifier, Misc

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, IndexOpInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `pcRanges` | ::mlir::ArrayAttr | 64-bit float array attribute
| `normRanges` | ::mlir::ArrayAttr | 64-bit float array attribute
| `voxelSizes` | ::mlir::ArrayAttr | 64-bit float array attribute
| `maxVoxelNum` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `maxPointsPerVoxel` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `normDims` | ::mlir::ArrayAttr | 64-bit integer array attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `points` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `voxels` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type
| `coords` | tensor of 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer values or none type


## `hbir.point_pillar_scatter` (::mlir::hbdk::hbir::PointPillarScatterOp)

HBIR point pillar scatter op.

HBIR point pillar scatter.
Traits: CommonVerifier, Misc

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, IndexOpInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `outShape` | ::mlir::ArrayAttr | 64-bit integer array attribute with exactly 4 elements

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `voxels` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type
| `coords` | tensor of 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type


## `hbir.pow` (::mlir::hbdk::hbir::PowOp)

HBIR tensor pow.

Pow takes input data(lhs) and exponent Tensor(rhs),
                        and produces one output data where the function $f(x) = x ^ {exponent}$,
                        is applied to the data tensor element - wise
Traits: Broadcastable, CommonVerifier, EltwiseLike

Interfaces: DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `lhs` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type
| `rhs` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type


## `hbir.prelu` (::mlir::hbdk::hbir::PreluOp)

HBIR tensor prelu

Return tensor after the operation, which has the same shape as the input.
Traits: CommonVerifier, Misc

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type
| `slope` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.relu` (::mlir::hbdk::hbir::ReLUOp)

HBIR ReLU activation.

Applies the rectified linear unit function element - wise.Relu function is defined as:

$ ReLU(x) = (x) ^ + = max(0, x) $

------

Shape:
* Input(*), where * means any number of dimensions.
* Output(*), same shapes as the input.

------

Prototype: Pytorch Relu.

Traits: CommonVerifier, LutLike, NaiveRoiInfer, NaiveTiling, SameElementType, SameOperandsAndResultShape

Interfaces: CalibOp, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), NonBatchAxesInfer, Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.reciprocal` (::mlir::hbdk::hbir::ReciprocalOp)

HBIR tensor reciprocal.

Applies reciprocal operator element - wise, $y = \frac{1}{x}$.

------

Note:
* Returns a new tensor with the reciprocal of the elements of input.


Traits: CommonVerifier, EltwiseLike, SameOperandsAndResultShape

Interfaces: DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.reduce_all` (::mlir::hbdk::hbir::ReduceAllOp)

Tests if all elements in input evaluate to True.

Return True if all elements in the row evaluate to True and False otherwise.

------

Parameters:
* input: the input tensor.
* dims: dimensions to perform reduce all on. If it's a list, reduce over all of them. Accepted range is [-r, r - 1] where r = rank(input).
* keepDim: keep the reduced dimensions or not. Default true means keep reduced dimensions.

------

Shape:
* input: $(*)$, where * represents any dimension.
* output: if keepDim is True, same as input. Otherwise, all reduced dims will be discarded (e.g. if input is of shape 1x2x3x4 and dims=[1, 2], the output shape will be 1x4).


Traits: CommonVerifier, ReduceLike

Interfaces: DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, RoiInfer, SchedInterface, SchedTemp, ShapeInference, Tiling

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `dims` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `keepDim` | ::mlir::BoolAttr | bool attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or  or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of  values or none type


## `hbir.reduce_argmax` (::mlir::hbdk::hbir::ReduceArgmaxOp)

Calculate max on multiple axes and return its index.

Return the indices of the max elements of the input tensor's element along the provided axis.

------

Parameters:
* input: the input tensor.
* dims: dimension to perform reduce argmax on. Accepted range is [-r, r - 1] where r = rank(input).
* keepDim: keep the reduced dimensions or not. Default true means keep reduced dimensions.

------

Shape:
* input: $(*)$, where * represents any dimension.
* output: if keepDim is True, same as input. Otherwise, all reduced dims will be discarded (e.g. if input is of shape 1x2x3x4 and dims=[1], the output shape will be 1x3x4).

------

Prototype: ONNX ReduceArgMax.

Traits: CommonVerifier, MoveF16CastLike, ReduceLike

Interfaces: DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, IndexOpInterface, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, RoiInfer, SchedInterface, SchedTemp, ShapeInference, Tiling

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `dims` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `keepDim` | ::mlir::BoolAttr | bool attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer values or none type


## `hbir.reduce_argmin` (::mlir::hbdk::hbir::ReduceArgminOp)

Calculate min on multiple axes and return its index.

Return the indices of the min elements of the input tensor's element along the provided axis.

------

Parameters:
* input: the input tensor.
* dims: dimension to perform reduce argmin on. Accepted range is [-r, r - 1] where r = rank(input).
* keepDim: keep the reduced dimensions or not. Default true means keep reduced dimensions.

------

Shape:
* input: $(*)$, where * represents any dimension.
* output: if keepDim is True, same as input. Otherwise, all reduced dims will be discarded (e.g. if input is of shape 1x2x3x4 and dims=[1], the output shape will be 1x3x4).

------

Prototype: ONNX ReduceArgMin.

Traits: CommonVerifier, MoveF16CastLike, ReduceLike

Interfaces: DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, IndexOpInterface, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, RoiInfer, SchedInterface, SchedTemp, ShapeInference, Tiling

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `dims` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `keepDim` | ::mlir::BoolAttr | bool attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer values or none type


## `hbir.reduce_max` (::mlir::hbdk::hbir::ReduceMaxOp)

Calculate max on multiple axes.

Return the max value of all elements in the provided axes of the input tensor.

------

Parameters:
* input: the input tensor.
* dims: dimensions to perform reduce max on. If it's a list, reduce over all of them. Accepted range is [-r, r - 1] where r = rank(input).
* keepDim: keep the reduced dimensions or not. Default true means keep reduced dimensions.

------

Shape:
* input: $(*)$, where * represents any dimension.
* output: if keepDim is True, same as input. Otherwise, all reduced dims will be discarded (e.g. if input is of shape 1x2x3x4 and dims=[1, 2], the output shape will be 1x4).

------

Prototype: ONNX ReduceMax.

Traits: CommonVerifier, MoveF16CastLike, ReduceLike

Interfaces: DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, RoiInfer, SchedInterface, SchedTemp, ShapeInference, Tiling

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `dims` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `keepDim` | ::mlir::BoolAttr | bool attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.reduce_mean` (::mlir::hbdk::hbir::ReduceMeanOp)

Calculate mean on multiple axes.

Return the mean of all elements in the provided axes of the input tensor.

------

Parameters:
* input: the input tensor.
* dims: dimensions to perform reduce mean on. If it's a list, reduce over all of them. Accepted range is [-r, r - 1] where r = rank(input).
* keepDim: keep the reduced dimensions or not. Default true means keep reduced dimensions.

------

Shape:
* input: $(*)$, where * represents any dimension.
* output: if keepDim is True, same as input. Otherwise, all reduced dims will be discarded (e.g. if input is of shape 1x2x3x4 and dims=[1, 2], the output shape will be 1x4).

------

Prototype: ONNX ReduceMean.

Traits: CommonVerifier, ReduceLike

Interfaces: DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, RoiInfer, SchedInterface, SchedTemp, ShapeInference, Tiling

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `dims` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `keepDim` | ::mlir::BoolAttr | bool attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.reduce_min` (::mlir::hbdk::hbir::ReduceMinOp)

Calculate min on multiple axes.

Return the min value of all elements in the provided axes of the input tensor.

------

Parameters:
* input: the input tensor.
* dims: dimensions to perform reduce min on. If it's a list, reduce over all of them. Accepted range is [-r, r - 1] where r = rank(input).
* keepDim: keep the reduced dimensions or not. Default true means keep reduced dimensions.

------

Shape:
* input: $(*)$, where * represents any dimension.
* output: if keepDim is True, same as input. Otherwise, all reduced dims will be discarded (e.g. if input is of shape 1x2x3x4 and dims=[1, 2], the output shape will be 1x4).

------

Prototype: ONNX ReduceMin

Traits: CommonVerifier, MoveF16CastLike, ReduceLike

Interfaces: DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, RoiInfer, SchedInterface, SchedTemp, ShapeInference, Tiling

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `dims` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `keepDim` | ::mlir::BoolAttr | bool attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.reduce_sum` (::mlir::hbdk::hbir::ReduceSumOp)

Calculate sum on multiple axes.

Return the sum of all elements in the provided axes of the input tensor.

------

Parameters:
* input: the input tensor.
* dims: dimensions to perform reduce sum on. If it's a list, reduce over all of them. Accepted range is [-r, r - 1] where r = rank(input).
* keepDim: keep the reduced dimensions or not. Default true means keep reduced dimensions.

------

Shape:
* input: $(*)$, where * represents any dimension.
* output: if keepDim is True, same as input. Otherwise, all reduced dims will be discarded (e.g. if input is of shape 1x2x3x4 and dims=[1, 2], the output shape will be 1x4).

------

Prototype: ONNX ReduceSum.

Traits: CommonVerifier, ReduceLike

Interfaces: DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, RoiInfer, SchedInterface, SchedTemp, ShapeInference, Tiling

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `dims` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `keepDim` | ::mlir::BoolAttr | bool attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.reshape` (::mlir::hbdk::hbir::ReshapeOp)

View a tensor as another shape

Returns a tensor with the same data and number of elements as input, but with the specified shape.When possible, the returned tensor will be a view of input.

------

Note:
* shape - the new shape.

------

Prototype: Pytorch reshape.

Traits: CommonVerifier, Foldable, MoveF16CastLike, MoveLike, SameElementType

Interfaces: DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, Layout, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), NonBatchAxesInfer, Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `shape` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `foldable` | ::mlir::BoolAttr | bool attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type


## `hbir.resize2d` (::mlir::hbdk::hbir::Resize2dOp)

HBIR 2-D resizing.

Scale the input proportionally.

------

Note:
* ratio controls zoom size.*mode controls interpolation type, it's default value is nearest.

------

Shape:
* Input: $(*, H_{in}, W_{in}, C)$
* Output: $(*, H_{out}, W_{out}, C)$, where

$ H_{out} = \lfloor{H_{in} * ratio}\rfloor $

$ W_{out} = \lfloor{W_{in} * ratio}\rfloor $

------

Prototype: Pytorch upsample_nearest2d.

Traits: CommonVerifier, Expansion, Round, SampleLike

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `ratio` | ::mlir::ArrayAttr | 64-bit float array attribute with exactly 2 elements
| `size` | ::mlir::ArrayAttr | 64-bit integer array attribute with exactly 2 elements
| `step` | ::mlir::ArrayAttr | 64-bit float array attribute with exactly 2 elements
| `initialOffset` | ::mlir::ArrayAttr | 64-bit float array attribute with exactly 2 elements
| `mode` | ::mlir::hbdk::InterpolationModeAttr | interpolation mode for all march
| `expansionMode` | ::mlir::hbdk::ExpansionModeAttr | mode to expand input feature on H/W
| `padValue` | ::mlir::Attribute | 64-bit float attribute or 64-bit signless integer attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type


## `hbir.rle` (::mlir::hbdk::hbir::RleOp)

HBIR rle op

Run length encode along with writing data from L1M to Ddr.
Traits: CommonVerifier, Misc

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type


## `hbir.roi_align` (::mlir::hbdk::hbir::RoiAlignOp)

HBIR roi align.

Crop the ROIs from a few feature maps, and resize them to the specified shape.The first input represents the ROI,
and the rest of the inputs serve as features.

------

Shape:
* Inputs: this first input is Roi: $(N, 4)$ or $(N, 6)$ represent the N ROIs, the rest of the inputs is input features: $(H_{in}, W_{in}, C_{in})$
* Output: $(N, H_{out}, W_{out}, C_{in})$,
where $H_{out}$ and $W_{out}$ are the specified shape controled by the parameter.

Traits: CommonVerifier, Misc

Interfaces: AdditionalMemoryRequire, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, Layout, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `shape` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `featureStrides` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `samplingRatio` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `interpolateMode` | ::mlir::StringAttr | string attribute
| `canonicalBoxSize` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `canonicalLevel` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `boxClipRatio` | ::mlir::ArrayAttr | 64-bit float array attribute with exactly 4 elements

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `inputs` | variadic of tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values


## `hbir.roll` (::mlir::hbdk::hbir::RollOp)

Roll the tensor along the given dimensions

Roll the tensor along the given dimension.Elements that are shifted beyond the last position are re-introduced at the first position.

------

Note:
* shifts - The number of places by which the elements of the tensor are shifted.
* dims -Axis along which to roll.

------
Prototype: Pytorch roll.

Traits: CommonVerifier, MoveF16CastLike, MoveLike, SameElementType

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, RoiInfer, SchedInterface, SchedTemp, ShapeInference, Tiling

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `shifts` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `dims` | ::mlir::ArrayAttr | 64-bit integer array attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type


## `hbir.round` (::mlir::hbdk::hbir::RoundOp)

HBIR tensor round.

Rounds elements of input to the nearest integer.

------

Note:
* This function implements the round half to even.
* Returns a new tensor with the round of the elements of input.


Traits: CommonVerifier, EltwiseLike, SameOperandsAndResultShape

Interfaces: DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `decimals` | ::mlir::IntegerAttr | 64-bit signless integer attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.rpp_v2` (::mlir::hbdk::hbir::RppV2Op)

HBIR RCNN post process.Real useful version

Process a few feature maps, generating a few boxes(ROIs).
Traits: CommonVerifier, Misc

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, NoMemoryEffect (MemoryEffectOpInterface), Perf, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `original_img_h` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `original_img_w` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `nms_threshold` | ::mlir::FloatAttr | 64-bit float attribute
| `score_threshold` | ::mlir::FloatAttr | 64-bit float attribute
| `class_number` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `nms_top_n` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `bbox_delta_mean` | ::mlir::ArrayAttr | 64-bit float array attribute
| `bbox_delta_std` | ::mlir::ArrayAttr | 64-bit float array attribute
| `image_size_fixed` | ::mlir::BoolAttr | bool attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `bbox_data` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type
| `bbox_score` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type values or none type
| `bbox_deltas` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `int_output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type
| `float_output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.rsqrt` (::mlir::hbdk::hbir::RsqrtOp)

HBIR tensor rsqrt.

Return tensor after the operation, which has the same shape as the input.
Traits: CommonVerifier, LutLike, SameOperandsAndResultShape

Interfaces: DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.scatter_elements` (::mlir::hbdk::hbir::ScatterElementsOp)

HBIR scatter elements op. Same semantics as scatter elements in onnx. In addition, it supports the mean mode of torch scatter_reduce

HBIR scatter elements scatter.The ONNX op link:
https: // github.com/onnx/onnx/blob/main/docs/Operators.md#ScatterElements

Copy the data to output, the specify a direction axis, use the values in updates to update the values in output at specific location according to indices.

In addition, it supports the mean mode of torch scatter_reduce.

Traits: CommonVerifier, Misc

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, IndexOpInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, RandomAccessInterface, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `axis` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `scatterReduceMode` | ::mlir::hbdk::ScatterReduceModeAttr | scatter reduce mode

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type
| `indices` | tensor of 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer values or none type
| `updates` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type


## `hbir.scatter_nd` (::mlir::hbdk::hbir::ScatterNDOp)

HBIR scatterND op. Same semantics as scatterND in onnx.

HBIR scatterDN.The ONNX op link:
https: // github.com/onnx/onnx/blob/main/docs/Operators.md#ScatterND

Copy the data to output, then use the values in updates to update the values in the output at some directions given by the indices.

Traits: CommonVerifier, Misc

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, IndexOpInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, RandomAccessInterface, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `scatterReduceMode` | ::mlir::hbdk::ScatterReduceModeAttr | scatter reduce mode

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `data` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values
| `indices` | tensor of 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer values
| `updates` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values


## `hbir.select` (::mlir::hbdk::hbir::SelectOp)

select a tensor from a bigger tensor on a specific dim and index

Slices the input tensor along the selected dimension at the given index.

------

Note:
* dim - the dimension to slice.
* index - the index to select with.
* Select operator is equivalent to slicing.

------

Prototype:Pytorch select.

Traits: CommonVerifier, MoveF16CastLike, MoveLike, SameElementType

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, RoiInfer, SchedInterface, SchedTemp, ShapeInference, Tiling

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `dim` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `index` | ::mlir::IntegerAttr | 64-bit signless integer attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type


## `hbir.sigmoid` (::mlir::hbdk::hbir::SigmoidOp)

HBIR Sigmoid activation.

Applies the element - wise function.Sigmoid function is defined as:

$ Sigmoid(x) = \frac{1} {1 + exp(-x)} $

------

Shape:
* Input: (*), where * means any number of dimensions.
* Output: (*), same shape as the input.

------

Prototype: Pytorch sigmoid.

Traits: CommonVerifier, LutLike, NaiveRoiInfer, NaiveTiling, SameOperandsAndResultShape

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.sign` (::mlir::hbdk::hbir::SignOp)

HBIR tensor sign.

Return tensor after the operation, which has the same shape as the input.
Traits: CommonVerifier, EltwiseLike, MoveF16CastLike, SameOperandsAndResultShape

Interfaces: DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.sin` (::mlir::hbdk::hbir::SinOp)

HBIR tensor sin.

Return tensor after the operation, which has the same shape as the input.
Traits: CommonVerifier, LutLike, SameOperandsAndResultShape

Interfaces: DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.sinh` (::mlir::hbdk::hbir::SinhOp)

HBIR tensor sinh.

Return tensor after the operation, which has the same shape as the input.
Traits: CommonVerifier, LutLike, SameOperandsAndResultShape

Interfaces: DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.slice` (::mlir::hbdk::hbir::SliceOp)

Slice a tensor out of a tensor

Slicing like python's style means taking elements from one given index to another given index.

------

Note:
* begin -the index start to pick(inclusive).
* end -the index end to pick(exclusive).
* step -the step interval of picking.

------

Prototype: Pytorch slice.

Traits: CommonVerifier, Foldable, MoveF16CastLike, MoveLike, SameElementType

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, Layout, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), NonBatchAxesInfer, Perf, Quantizable, RoiInfer, SchedInterface, SchedTemp, ShapeInference, Tiling

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `begin` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `end` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `step` | ::mlir::ArrayAttr | 64-bit integer array attribute
| `foldable` | ::mlir::BoolAttr | bool attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type


## `hbir.slice_scatter` (::mlir::hbdk::hbir::SliceScatterOp)

Embeds the values of the src tensor into input at the given dimension

Embeds the values of the src tensor into input at the given dimension. This function returns a tensor with fresh storage; it does not create a view.

------

Note:
* src (Tensor) -the tensor to embed into input.
* dim (int) -the dimension to insert the slice into.
* start (int) -the start index of where to insert the slice.
* end (int) -the end index of where to insert the slice.
* step (int) -the how many elements to skip in.

------

Prototype: Pytorch slice_scatter.

Traits: CommonVerifier, Misc

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `dim` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `start` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `end` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `step` | ::mlir::IntegerAttr | 64-bit signless integer attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type
| `src` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type


## `hbir.softmax` (::mlir::hbdk::hbir::SoftmaxOp)

HBIR Softmax Op.

Applies the Softmax function to an n - dimensional input Tensor rescaling them so that the elements of the n-dimensional output.
Tensor lie in the range $[0, 1] $ and sum to 1.

Softmax function is defined as:

$ Softmax(x_i) = \frac{exp(x_i)} {\sum_jexp(x_j)} $

------

Shape:
* Input(*), where $*$ means, any number of additional dimensions.*Output(*), same shape as the input.

------

Prototype: Pytorch softmax.

Traits: CommonVerifier, Misc, SameOperandsAndResultShape

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), NonBatchAxesInfer, Perf, Quantizable, RoiInfer, SchedInterface, SchedTemp, ShapeInference, Tiling

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `dim` | ::mlir::IntegerAttr | 64-bit signless integer attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.softplus` (::mlir::hbdk::hbir::SoftplusOp)

HBIR Softplus Op.

Applies the SoftPlus function element-wise. SoftPlus function defined as:

$ SoftPlus(x) = \frac{1}{\beta}*log(1+exp(\beta * x)) $

SoftPlus is  a smooth approximation to the ReLU function.

------

Note:
* beta - the $\beta$ value for the Softplus formulation.
* max - values is for numerical stability, when $\beta *x > max, SoftPlus(x)=x$.

------

Shape:
* Input: (*), where * means any number of dimensions.
* Output: (*), same shape as the input.

------

Prototype: Pytorch softplus.

Traits: CommonVerifier, LutLike, NaiveRoiInfer, NaiveTiling, SameOperandsAndResultShape

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `beta` | ::mlir::FloatAttr | 64-bit float attribute
| `threshold` | ::mlir::FloatAttr | 64-bit float attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.sort` (::mlir::hbdk::hbir::SortOp)

HBIR tensor sort

Sorts the elements of the input tensor along a given dimension in ascending order by value.

------

Parameters:
* input: the input tensor.
* dim: the dimension to sort along.
* descending: controls the sorting order (ascending or descending).
* stable: makes the sorting routine stable, which guarantees that the order of equivalent elements is preserved.

------

Shape:
* Input: $(N, *)$
* Values: $(N, *)$ (same shape as input)
* Indices: $(N, *)$ (same shape as input)

------

Prototype: Pytorch sort.

Traits: CommonVerifier, Misc

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, IndexOpInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `dim` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `descending` | ::mlir::BoolAttr | bool attribute
| `stable` | ::mlir::BoolAttr | bool attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `values` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type
| `indices` | tensor of 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer values or none type


## `hbir.sqrt` (::mlir::hbdk::hbir::SqrtOp)

HBIR tensor sqrt.

Applies square root operator element - wise, $y = \sqrt{x}$.

------

Note:
* Returns a new tensor with the square root of the elements of input. If input is negative, then it will return NaN.


Traits: CommonVerifier, LutLike, SameOperandsAndResultShape

Interfaces: DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.stack` (::mlir::hbdk::hbir::StackOp)

Stack multiple tensors along one extra dimension

Concatenates a sequence of tensors along a new dimension.No elemental type conversion.

------

Note:
* All tensors need to be of the same size.

------

Prototype: Pytorch stack.

Traits: CommonVerifier, MoveF16CastLike, MoveLike, SameElementType

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `dim` | ::mlir::IntegerAttr | 64-bit signless integer attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `inputs` | variadic of tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type


## `hbir.sub` (::mlir::hbdk::hbir::SubOp)

HBIR tensor substraction.

Applies substraction operator element-wise, $y_i=lhs_i-rhs_i$.

------

Note:
* Our arithmetic operator support broadcast.

Traits: Broadcastable, CommonVerifier, EltwiseLike

Interfaces: CalibOp, DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `lhs` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type
| `rhs` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type


## `hbir.swish` (::mlir::hbdk::hbir::SwishOp)

HBIR Swish activation.

Applies the swish function element-wise.Swish function defined as:

$ Swish(x) = x * Sigmoid(x) $

------

Shape:
* Input: (*), where * means any number of dimensions.
* Output: (*), same shape as the input.

------

Prototype: Pytorch silu.

Traits: CommonVerifier, LutLike, NaiveRoiInfer, NaiveTiling, SameOperandsAndResultShape

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.tan` (::mlir::hbdk::hbir::TanOp)

HBIR tensor tan.

Return tensor after the operation, which has the same shape as the input.
Traits: CommonVerifier, LutLike, SameOperandsAndResultShape

Interfaces: DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.tanh` (::mlir::hbdk::hbir::TanhOp)

HBIR Tanh activation.

Applies the tanh function element - wise.Tanh function is defined as:

$ Tanh(x) =\frac{exp(x) - exp(-x)} {exp(x) + exp(-x)} $

------

Shape:
* Input: (*), where * means any number of dimensions.
* Output: (*), same shape as the input.

------

Prototype: Pytorch tanh.

Traits: CommonVerifier, LutLike, NaiveRoiInfer, NaiveTiling, SameOperandsAndResultShape

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, MoveTransposeInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.tile` (::mlir::hbdk::hbir::TileOp)

Constructs a tensor by tiling a given tensor.


Traits: CommonVerifier, MoveF16CastLike, MoveLike, SameElementType

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `multiplies` | ::mlir::ArrayAttr | 64-bit integer array attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type


## `hbir.topk` (::mlir::hbdk::hbir::TopkOp)

HBIR tensor topk

Returns the k largest elements of the given input tensor along a given dimension.

If dim is not given, the last dimension of the input is chosen.

If largest is False then the k smallest elements are returned.

values, indices are returned in separate tensors, where the indices are the indices of the elements in the original input tensor.

The boolean option sorted if True, will make sure that the returned k elements are themselves sorted.

Traits: CommonVerifier, Misc, MoveF16CastLike

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, IndexOpInterface, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `k` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `dim` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `largest` | ::mlir::BoolAttr | bool attribute
| `sorted` | ::mlir::BoolAttr | bool attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `values` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type
| `indices` | tensor of 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer values or none type


## `hbir.transpose` (::mlir::hbdk::hbir::TransposeOp)

Reverse or permute the dims of an array; returns the modified array.

Returns a tensor that is a view of the original tensor input with its dimesions permuted.

------

Note:
* input: the input tensor.
* dims: the desired ordering of dimensions.

------

Prototype: Pytorch permute.

Traits: CommonVerifier, MoveF16CastLike, MoveLike, SameElementType

Interfaces: DataMovePropagate, HBTLExecutable, HbdkExecutorInterface, HbdkInferType, Layout, NoMemoryEffect (MemoryEffectOpInterface), NonBatchAxesInfer, Perf, Quantizable, RoiInfer, SchedInterface, SchedTemp, ShapeInference, Tiling

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `dims` | ::mlir::ArrayAttr | 64-bit integer array attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values


## `hbir.warp` (::mlir::hbdk::hbir::WarpOp)

HBIR warp.

From the input, sample(bi - linear interpolation) pixels specified by grid.

------

Shape:
* input: input of shape $(*, H_{in}, W_{in}, C_{in})$, where * represent any number of dimension.
* grid: flow - field of shape $(*, H_{out}, W_{out}, 2)$
* output: $(*, H_{out}, W_{out}, C_{in})$

Traits: CommonVerifier, Expansion, Round, SampleLike

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, RandomAccessInterface, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `mode` | ::mlir::hbdk::InterpolationModeAttr | interpolation mode for all march
| `expansionMode` | ::mlir::hbdk::ExpansionModeAttr | mode to expand input feature on H/W
| `padValue` | ::mlir::Attribute | 64-bit float attribute or 64-bit signless integer attribute

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type
| `move` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  values or none type


## `hbir.where` (::mlir::hbdk::hbir::WhereOp)

HBIR where op

Return a tensor of elements selected from either lhs or rhs, depending on condition.

Traits: Broadcastable, CommonVerifier, EltwiseLike

Interfaces: HBTLExecutable, HbdkExecutorInterface, HbdkInferType, NoMemoryEffect (MemoryEffectOpInterface), Perf, Quantizable, SchedInterface, SchedTemp, ShapeInference

Effects: MemoryEffects::Effect{}

### Operands:

| Operand | Description |
| :-----: | ----------- |
| `condition` | tensor of  values or none type
| `lhs` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type
| `rhs` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type

### Results:

| Result | Description |
| :----: | ----------- |
| `output` | tensor of 64-bit float or 32-bit float or 16-bit float or bfloat16 type or 8-bit signed integer or 16-bit signed integer or 32-bit signed integer or 64-bit signed integer or 8-bit unsigned integer or 16-bit unsigned integer or 32-bit unsigned integer or 64-bit unsigned integer or  or  values or none type


