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

实现 **推理/LLM 相关**算子时，应主动对照下列仓库中的 **目录结构、launch 封装、与 PyTorch 的绑定方式**，避免闭门造车。除 **vLLM**、**TensorRT-LLM** 外，建议将 **SGLang**、**FlashAttention**、**FlashInfer**、**DeepSeek 官方开源**、**Transformer Engine / Apex**、**AWQ / GPTQ / SmoothQuant / llm-compressor** 及 **CUDA Graph** 用法等纳入检索范围（下列为高频对照入口，版本以各仓库默认分支为准）。

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

### NVIDIA 训练侧数值库（FP8 / 融合 / 分布式辅助）

| 项目 | 典型参考价值 | URL |
|------|----------------|-----|
| **Transformer Engine** | **FP8 / BF16** 下 LayerNorm、GEMM、Attention 等与 **TE 模块** 的融合与 scaling 约定；与 Megatron/NeMo 等栈对齐时的 kernel 与 API 契约 | https://github.com/NVIDIA/TransformerEngine |
| **Apex** | 分布式与 **fused optimizers / 归一化 / 激活** 等（部分场景已被 TE / 原生 PyTorch 替代）；遗留代码路径与 launch 封装仍可对照 | https://github.com/NVIDIA/apex |

### 权重量化与压缩（AWQ / GPTQ / SmoothQuant / llm-compressor）

实现 **W4A16、INT8、混合精度** 等推理算子时，应对照 **校准流程、scale/zero-point、GEMM/GEMV kernel** 与部署栈一致性：

| 项目 | 典型参考价值 | URL |
|------|----------------|-----|
| **LLM Compressor** | 与 **vLLM 生态** 衔接的压缩/稀疏/量化工具链；导出与 serving 对齐方式 | https://github.com/vllm-project/llm-compressor |
| **AWQ** | 激活感知的权重量化与内核/实现参考 | https://github.com/mit-han-lab/llm-awq |
| **AutoAWQ** | AWQ 推理与 **GEMM 实现**、Python 绑定组织 | https://github.com/casper-hansen/AutoAWQ |
| **GPTQ**（算法参考） | GPTQ 量化流程与误差度量 | https://github.com/IST-DASlab/gptq |
| **AutoGPTQ** | GPTQ 生态实现与 **CUDA extension** 布局 | https://github.com/AutoGPTQ/AutoGPTQ |
| **SmoothQuant** | 训练后 INT8 等：**平滑因子**、与 GEMM 融合的数值路径 | https://github.com/mit-han-lab/smoothquant |

### 运行时：CUDA Graph（非独立 Git 仓库）

**CUDA Graph** 将多条 kernel 与 memcpy **捕获为一张图** 以降低 launch 开销；自定义算子若落在 Graph 路径上，需在设计与性能阶段显式考虑：

- **CUDA C++ 编程指南** — Graph 语义、capture 限制、whole-graph launch：<https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs>
- **PyTorch** — `torch.cuda.CUDAGraph`、`torch.compile` / Inductor 与 cudagraphs 相关选项；**静态 shape、避免非法同步、内存池（如 graph-safe allocator）** 与自定义 op 的兼容性需在性能报告中说明。

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

在 **vLLM / TensorRT-LLM / SGLang / FlashAttention / FlashInfer / deepseek-ai / Triton / Transformer Engine / llm-compressor / AWQ·GPTQ 生态** 各子仓及 **CUDA Graph 文档** 中定位同类实现时，先查 [reference-target-models.md](reference-target-models.md) §4–§5，再在仓库内搜索并写入设计中的 **具体路径或文件名**（路径随版本变化，以检出结果为准）：

| 模型族 | 优先对照仓库 | 说明 |
|--------|--------------|------|
| Qwen | vLLM、TensorRT-LLM、**SGLang**、**FlashInfer**、PyTorch `native/cuda`、**FlashAttention**、**llm-compressor**、**AutoAWQ/AutoGPTQ**（若走权重量化） | Attention/RoPE/Norm/FFN、decode kernel、量化与压缩路径 |
| DeepSeek（含 MoE） | 同上 + **deepseek-ai**（如 DeepSeek-V3、**DeepGEMM**）+ **Transformer Engine**（若训练/推理走 TE-FP8）+ 关键词 `moe` / `expert` / `router` / `grouped` | 官方与社区栈交叉验证；分组 GEMM、路由、FP8；**CUDA Graph** 与多 kernel 融合时的捕获边界 |

## 设计阶段要写下来

- 本算子是否已被 **cuBLAS/CUTLASS/cuDNN** 覆盖？覆盖则设计 **API 选型 + workspace + stream + dtype**。
- 若必须手写 kernel：一句话说明 **库无法覆盖的原因**（layout、融合方式、非标准归约等）。
- 列出 **要对齐的参考路径**（例如 `vllm/...`、`tensorrt_llm/...`、`sglang/...`、`flash-attention/...`、`flashinfer/...`、`DeepGEMM/...`、`triton/...`、`TransformerEngine/...`、`apex/...`、`llm-compressor/...`、`autoawq/...`、`AutoGPTQ/...`、`smoothquant/...` 下的具体子目录或文件名，便于 coder 检索）。
- 若使用 **CUDA Graph**：在设计/性能中写明 **capture 范围**、是否与 **PyTorch CUDAGraph** 或 **纯 CUDA Graph API** 结合、**静态约束**（shape、同步、allocator）。
- 若目标为 **Qwen / DeepSeek**：在设计中交叉引用 [reference-target-models.md](reference-target-models.md) 中的算子条目与精度默认。
