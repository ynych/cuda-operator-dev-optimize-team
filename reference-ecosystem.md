# CUDA 生态与参考实现（优先复用）

**原则**：能用成熟库与开源实现就不要从零写 `__global__`。自定义算子常见形态是 **薄封装 + 正确 launch/Workspace**，或 **CUTLASS 模板组合**，而不是手写 GEMM/规约全家桶。

## 官方 / NVIDIA 栈（优先）

| 能力 | 首选依赖 | 说明 |
|------|----------|------|
| GEMM/Batched GEMM | **cuBLAS** / **cuBLASLt** | 列主序、stride、Epilogue 先读文档再封装 |
| 高性能 GEMM 模板 | **CUTLASS** | 复杂 layout、融合、多 arch 时优先于手写 WMMA |
| 卷积等 DL 原语 | **cuDNN** | 与框架 adapter 对齐 |
| 规约、scan、sort、并行原语 | **CUB**（CCCL 一部分） | 与 Thrust 同生态 |

- CUTlass: https://github.com/NVIDIA/cutlass  
- cuBLAS: https://docs.nvidia.com/cuda/cublas/  
- CCCL（CUB/Thrust）: https://github.com/NVIDIA/cccl  

## 开源框架中的实现范式（作示例，非依赖项）

实现 **推理/LLM 相关**算子时，应主动对照下列仓库中的 **目录结构、launch 封装、与 PyTorch 的绑定方式**，避免闭门造车：

| 项目 | 典型参考价值 | URL |
|------|----------------|-----|
| **vLLM** | PagedAttention、KV cache、自定义 CUDA extension 与 Python 绑定组织方式 | https://github.com/vllm-project/vllm |
| **TensorRT-LLM** | 融合 MHA、GEMM/量化、插件式 kernel 与构建系统集成 | https://github.com/NVIDIA/TensorRT-LLM |

其他常作对照：**FlashAttention**（https://github.com/Dao-AILab/flash-attention）、**xFormers**、各框架内置 `aten/src/ATen/native/cuda`。

## 按目标模型族检索（Qwen / DeepSeek）

**默认目标模型族**为 **Qwen** 与 **DeepSeek** 推理栈。算子级别的关注点、MoE、量化与精度默认见 **[reference-target-models.md](reference-target-models.md)**。

在 vLLM / TensorRT-LLM 中定位同类实现时，先查该文档 §4 关键词，再在仓库内搜索并写入设计中的 **具体路径或文件名**（路径随版本变化，以检出结果为准）：

| 模型族 | 优先对照仓库 | 说明 |
|--------|--------------|------|
| Qwen | vLLM、TensorRT-LLM、PyTorch `native/cuda` | Attention/RoPE/Norm/FFN 与量化路径 |
| DeepSeek（含 MoE） | 同上 + 关键词 `moe` / `expert` / `router` / `grouped` | Dense 与 MoE 分组 GEMM、路由 |

## 设计阶段要写下来

- 本算子是否已被 **cuBLAS/CUTLASS/cuDNN** 覆盖？覆盖则设计 **API 选型 + workspace + stream + dtype**。
- 若必须手写 kernel：一句话说明 **库无法覆盖的原因**（layout、融合方式、非标准归约等）。
- 列出 **要对齐的参考路径**（例如 `vllm/...`、`tensorrt_llm/...` 下的具体子目录或文件名，便于 coder 检索）。
- 若目标为 **Qwen / DeepSeek**：在设计中交叉引用 [reference-target-models.md](reference-target-models.md) 中的算子条目与精度默认。
