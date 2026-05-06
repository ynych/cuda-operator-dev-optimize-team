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

实现 **推理/LLM 相关**算子时，应主动对照下列仓库中的 **目录结构、launch 封装、与 PyTorch 的绑定方式**，避免闭门造车。除 **vLLM**、**TensorRT-LLM** 外，建议将 **SGLang**、**FlashAttention**、**FlashInfer**、**DeepSeek 官方开源** 等纳入检索范围（下列为高频对照入口，版本以各仓库默认分支为准）。

### 推理 / 服务栈

| 项目 | 典型参考价值 | URL |
|------|----------------|-----|
| **vLLM** | PagedAttention、KV cache、自定义 CUDA extension 与 Python 绑定组织方式 | https://github.com/vllm-project/vllm |
| **TensorRT-LLM** | 融合 MHA、GEMM/量化、插件式 kernel 与构建系统集成 | https://github.com/NVIDIA/TensorRT-LLM |
| **SGLang** | 高吞吐推理服务、RadixAttention、调度与 **backend / custom op** 组织方式；常与 **FlashInfer** 组合做 decode/attention kernel | https://github.com/sgl-project/sglang |

### 融合注意力与高性能 Kernel（影响力大、常作黄金参考）

| 项目 | 典型参考价值 | URL |
|------|----------------|-----|
| **FlashAttention**（含 FA2/FA3 演进） | IO-aware 融合注意力、tiling 与内核组织；自定义算子如何与 PyTorch 对接 | https://github.com/Dao-AILab/flash-attention |
| **FlashInfer** | 面向 **LLM 推理** 的 **CUDA kernel 集合**（含 page attention、decode/sampling 等）；与 SGLang 等栈集成方式、Python/C++ 绑定组织 | https://github.com/flashinfer-ai/flashinfer |
| **DeepGEMM**（DeepSeek 官方） | 面向 LLM 的 FP8 GEMM 等高性能 CUDA 实现与封装方式 | https://github.com/deepseek-ai/DeepGEMM |
| **DeepSeek-V3**（官方模型/推理参考） | 公开模型代码中与 MoE、推理路径相关的模块布局（与具体算子需求交叉检索） | https://github.com/deepseek-ai/DeepSeek-V3 |

**DeepSeek 组织入口**（按需下钻子仓库）：https://github.com/deepseek-ai  

### 算子 DSL / 编译生成（与手写 CUDA 对照）

| 项目 | 典型参考价值 | URL |
|------|----------------|-----|
| **Triton** | Python DSL 生成 GPU kernel；与 PyTorch custom op、`torch.compile` 生态衔接；适合对照 **tile 语义与内存层次** | https://github.com/triton-lang/triton |
| **PyTorch Inductor / torch.compile** | 算子如何落入 **generated Triton/CUDA**；与手写 extension 的边界与融合机会 | https://github.com/pytorch/pytorch（`torch/_inductor` 等） |

### 其他高影响力（按需检索）

| 项目 | 典型参考价值 | URL |
|------|----------------|-----|
| **Megatron-LM** | 大规模训练中的并行与张量布局、部分 fused kernel 与配置范式 | https://github.com/NVIDIA/Megatron-LM |
| **LMDeploy** | 推理部署与 **TurboMind** 等引擎中的 kernel/引擎组织 | https://github.com/InternLM/lmdeploy |
| **llama.cpp** | 跨后端推理参考（含 CUDA）；低比特量化与算子取舍 | https://github.com/ggerganov/llama.cpp |
| **bitsandbytes** | 训练/推理侧 **NF4/INT8** 等量化与 CUDA kernel 封装 | https://github.com/bitsandbytes-foundation/bitsandbytes |

其他常作对照：**xFormers**（https://github.com/facebookresearch/xformers）、各框架内置 `aten/src/ATen/native/cuda`。

## 按目标模型族检索（Qwen / DeepSeek）

**默认目标模型族**为 **Qwen** 与 **DeepSeek** 推理栈。算子级别的关注点、MoE、量化与精度默认见 **[reference-target-models.md](reference-target-models.md)**。

在 **vLLM / TensorRT-LLM / SGLang / FlashAttention / FlashInfer / deepseek-ai / Triton 生态** 各子仓中定位同类实现时，先查 [reference-target-models.md](reference-target-models.md) §4 关键词，再在仓库内搜索并写入设计中的 **具体路径或文件名**（路径随版本变化，以检出结果为准）：

| 模型族 | 优先对照仓库 | 说明 |
|--------|--------------|------|
| Qwen | vLLM、TensorRT-LLM、**SGLang**、**FlashInfer**、PyTorch `native/cuda`、**FlashAttention** | Attention/RoPE/Norm/FFN、decode kernel、量化路径 |
| DeepSeek（含 MoE） | 同上 + **deepseek-ai**（如 DeepSeek-V3、**DeepGEMM**）+ 关键词 `moe` / `expert` / `router` / `grouped` | 官方实现与社区栈交叉验证；分组 GEMM、路由、FP8 GEMM |

## 设计阶段要写下来

- 本算子是否已被 **cuBLAS/CUTLASS/cuDNN** 覆盖？覆盖则设计 **API 选型 + workspace + stream + dtype**。
- 若必须手写 kernel：一句话说明 **库无法覆盖的原因**（layout、融合方式、非标准归约等）。
- 列出 **要对齐的参考路径**（例如 `vllm/...`、`tensorrt_llm/...`、`sglang/...`、`flash-attention/...`、`flashinfer/...`、`DeepGEMM/...`、`triton/...` 下的具体子目录或文件名，便于 coder 检索）。
- 若目标为 **Qwen / DeepSeek**：在设计中交叉引用 [reference-target-models.md](reference-target-models.md) 中的算子条目与精度默认。
