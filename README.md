# CUDA Operator Dev & Optimize Team

## 项目定位

本仓库是一个 **团队型 Agent Skill（team-skill）**，用于在编排环境中驱动 **CUDA 自定义算子 / kernel** 的端到端交付：**设计 → 实现 → 对抗审查 → 精度验收 → 性能 profile**，并强制 **「NVIDIA 生态与开源参考优先」**（cuBLAS / cuBLASLt、CUTLASS、cuDNN、CUB 与 **vLLM、TensorRT-LLM、SGLang、FlashAttention、FlashInfer、Triton、Transformer Engine、Apex、llm-compressor、AWQ/GPTQ/SmoothQuant、CUDA Graph、DeepSeek 官方开源** 等范式），避免无必要地从零手写核心算子。

**默认目标工作负载**：以 **Qwen 系列** 与 **DeepSeek 系列** 大模型的推理（及相关的训练/对齐算子）为主。算子需求应优先与这两条模型线的 **典型结构（Attention、RoPE、RMSNorm、SwiGLU、MoE 等）**、**精度路径（bf16 / fp8 / 量化）** 对齐；其他模型族需在 Pre-flight 中显式声明或切换上下文。

## 适用场景

- 为 **Qwen / DeepSeek** 服务栈新增或优化 CUDA 扩展、融合 kernel、与 PyTorch / vLLM / SGLang 等绑定层。
- 需要 **可重复门禁**：设计文档含生态策略与目标模型对齐、代码可编译、对抗审查、golden 精度、Nsight 类性能证据。

## 不适用场景

- 无门禁的一次性脚本、与 CUDA 无关的纯 Python 逻辑。
- 拒绝提供 golden 或 GPU/profile 环境且不接受 **DEGRADED** 标注的「口头优化」。

## 文档结构

| 文件 | 说明 |
|------|------|
| [SKILL.md](SKILL.md) | Skill 入口：流程总览、角色索引、Pre-flight |
| [workflow.md](workflow.md) | 步骤、Mermaid、Final Report 模板 |
| [bind.md](bind.md) | 预算、行为约束、降级 |
| [dependencies.yaml](dependencies.yaml) | 工具与子 skill 预检 |
| [reference-ecosystem.md](reference-ecosystem.md) | NVIDIA 栈、推理栈、TE/Apex、量化（llm-compressor/AWQ/GPTQ/SmoothQuant）、CUDA Graph、DeepSeek 官方等参考 |
| [reference-target-models.md](reference-target-models.md) | **Qwen / DeepSeek** 算子关注点、MoE、量化与精度默认 |
| [roles/](roles/) | 五角色：设计、编码、对抗、精度、性能 |

## 如何触发本 Skill

编排器多按 `SKILL.md` 里 **`description`** 关键词匹配。若 **未自动选中**，请用户句子里显式包含：**CUDA / kernel / 算子 / 优化 / matrix transpose / `.cu` / M= N= / ref / golden** 等，或直接写：**「使用 cuda-operator-dev-optimize-team 团队 skill」**。

四类典型问法见 [SKILL.md](SKILL.md) 章节 **「如何触发（编排 / 用户话术）」**（从零、仅 shape、已有 solution、实现+ref）。

## 快速使用（编排侧）

1. 将本目录作为 **team-skill** 加载（见 `SKILL.md`  frontmatter）。
2. Leader **Step 0**：阅读 `dependencies.yaml`、`reference-ecosystem.md`、`reference-target-models.md`；确认目标是否为 Qwen / DeepSeek **或 generic 算子**（如转置：Target Model 填 other/generic）。
3. 按 `workflow.md` 串行派发五角色；子代理提示词使用各 `roles/*.md` 中的 **`## Inline Persona for Teammate`** 全文。

## 版本

与 [SKILL.md](SKILL.md) 中 `version` 字段保持一致。
