# 目标模型族：Qwen / DeepSeek（算子与验收对齐）

**用途**：Stage 1 设计、Stage 4 精度、Stage 5 性能报告时，将算子与 **默认服务模型族** 对齐。若任务针对其他模型，Leader 在 Pre-flight 中声明并替换本节默认值。

**原则**：不绑定某一具体 checkpoint 名称；以 **公开架构共性** 为纲，具体 **hidden/head/页大小/MoE 专家数** 由用户在需求或设计中给出。

---

## 1. 算子关注点速查（Dense 共性）

| 关注点 | 数学/结构角色 | 库与实现倾向 |
|--------|----------------|--------------|
| 注意力 | MHA/MQA/GQA、scale、mask、softmax | **FlashAttention** 系融合；**FlashInfer** 推理侧 attention/page 类 kernel；vLLM PagedAttention；SGLang RadixAttention / backend；TRT-LLM MHA 插件；CUTLASS/cuBLAS 作 GEMM 块 |
| RoPE | 位置嵌入与 cache 布局 | 与 KV layout、head 维对齐；对照框架 `native` 与 vLLM / SGLang / TRT-LLM 中 rope 相关实现 |
| RMSNorm / LayerNorm | 稳定归一 | cuDNN 或融合 kernel；手写时注意规约与数值 |
| FFN / SwiGLU | 门控前馈 | GEMM + 激活融合 → CUTLASS epilogue / cuBLASLt |
| 嵌入 / LM head | 大 vocab GEMM | cuBLASLt batched GEMM；量化时走量化 GEMM 路径 |
| KV Cache | layout、page、copy | vLLM `attention`/`cache`；SGLang 中 cache / attention backend；**FlashInfer** page/KV 相关实现；TRT-LLM KV 插件 |

详细库优先级见 [reference-ecosystem.md](reference-ecosystem.md)。

---

## 2. Qwen 系列（默认假设）

- **常见精度**：bf16 推理为主；部分路线 fp8 / int8 / GPTQ 等（见 §5）。
- **设计时写明**：hidden size、num heads / kv heads、head dim、是否 GQA/MQA、最大序列与是否 paged KV。
- **精度验收**：至少覆盖设计声明的 dtype；形状上含 **小 batch + 中长 seq** 与 **边界 seq（1、max）** 的代表组合（见 `roles/precision-validator.md` 默认模板）。

---

## 3. DeepSeek 系列与 MoE（§G）

- **Dense 部分**：与 §1 相同。
- **MoE 推理额外关注点**：
  - **路由**：top-k 专家选择、负载均衡相关的数值与稳定性（非性能优化不得改语义）。
  - **分组 GEMM**：多专家 batched GEMM / 稀疏聚集 → **cuBLASLt grouped GEMM / CUTLASS grouped** 优先于手写分块全家桶。
  - **通信与重叠**：若跨卡，性能阶段需区分 **纯 kernel 时间** 与 **端到端 step**（见 `roles/performance-optimizer.md` Serving 对齐）。
- **设计必选**：是否 MoE、`num_experts`、top-k、per-layer 专家映射；参考实现关键词 **MoE、expert、router、grouped GEMM** 在 vLLM / TensorRT-LLM / SGLang / **FlashInfer** / **deepseek-ai（DeepSeek-V3 等）** 仓库内交叉检索；FP8 GEMM 等可对照 **DeepGEMM**。

---

## 4. 参考实现检索关键词（多仓库）

版本迭代会导致路径变动，编码前在仓库内 **按关键词搜索** 并对照设计中的「参考路径」：

| 主题 | vLLM | TensorRT-LLM | SGLang | FlashAttention | FlashInfer | DeepSeek 官方（示例） |
|------|------|--------------|--------|----------------|------------|----------------------|
| 注意力 / KV | `attention`, `paged`, `kv_cache` | `attention`, `kv`, `context` | `attention`, `radix`, `kv`, `backend` | `flash_attn`, `cuda`, `src` | `page`, `decode`, `prefill`, `batch_attention` | `inference`, `model`, `attention`（按子仓调整） |
| RoPE | `rotary`, `rope` | `rotary`, `rope` | `rope`, `rotary` | （多在集成侧） | `rope`, `pos_enc` | `rope`, `embed` |
| LayerNorm / RMSNorm | `layernorm`, `rms` | `rmsnorm`, `norm` | `norm`, `layernorm` | — | `norm`, `rms` | `norm`, `rms` |
| MoE | `moe`, `expert`, `router` | `moe`, `expert`, `router` | `moe`, `expert` | — | （按版本检索） | `moe`, `expert`, `MoE` |
| 量化 / GEMM | `fp8`, `quant`, `awq`, `gptq` | `quantize`, `fp8`, `weight_only` | `fp8`, `quant`, `gemm` | — | `fp8`, `gemm`, `quant` | **DeepGEMM**：`gemm`, `fp8`, `mma` |

索引总表见 [reference-ecosystem.md](reference-ecosystem.md)「按目标模型族」一节。

---

## 5. 量化与数值路径（§H）

| 路径 | 设计注意 | 精度 golden | 性能注意 |
|------|----------|-------------|----------|
| bf16 / fp16 | 与训练对齐的累加顺序说明 | PyTorch ref 同 dtype；注意 `float32` acc 选项 | NCU 看 memory vs compute |
| FP8 (E4M3/E5M2 等) | 缩放、amax、与框架约定一致 | 需 **框架或 TRT-LLM / DeepGEMM 部署侧一致** 的 ref；atol/rtol 单独论证 | 对照 cuBLASLt FP8 API、TRT-LLM FP8 插件、**DeepGEMM** |
| INT8 / weight-only | 校准表、per-channel vs per-tensor | 与部署栈同一套 scale | 带宽与指令吞吐 |

**禁止**：在未更新设计的前提下，以「提速」为由更换量化scheme或缩放策略。

---

## 6. 精度验收默认建议（与 precision-validator 配合）

以下仅为 **起点**，最终以设计中的数值语义与框架 ref 为准：

| 场景 | 建议 rtol | 建议 atol | 备注 |
|------|-----------|-----------|------|
| bf16/fp16 非规约类逐元 | `1e-2` ~ `5e-2` | `1e-4` ~ `1e-3` | 大规约累加可放宽并说明理由 |
| fp32 严格对齐 | `1e-5` | `1e-6` | 作为 golden 链路的基准档 |
| FP8 / 整型量化 | **逐算子论证** | **逐算子论证** | 与部署 ref 绑定，不用 fp32 强行 allclose |

完整用例模板见 `roles/precision-validator.md`。
