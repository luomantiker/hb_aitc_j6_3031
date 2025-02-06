# 量化训练工具基础示例使用说明

## 流程示例

包含 FX Mode 使用流程（`run_pipeline/fx_mode.py`，推荐）和 Eager Mode 使用流程（`run_pipeline/eager_mode.py`）

请使用 `--help` 参数获取脚本使用说明

## 自定义算子示例

见 `custom_ops/`

### cpp 自定义算子

1. 编译自定义算子库
    ``` bash
    cd csrc
    bash compile.sh
    cd ..
    ```
2. 执行示例脚本 `custom_cpp_op.py`
