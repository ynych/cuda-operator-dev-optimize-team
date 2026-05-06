---
name: cuda-operator-dev-optimize-team
description: |
  5-role C+A pipeline for CUDA ops: prefer CUTLASS, cuBLAS, cuDNN, CUB before raw kernels; cites vLLM, TensorRT-LLM, SGLang, FlashAttention, FlashInfer, Triton, and deepseek-ai (e.g. DeepGEMM) as reference patterns.
  Default target workloads: Qwen and DeepSeek LLM serving (see reference-target-models.md). Use for end-to-end custom CUDA from spec to correct, profiled code. Not for trivial one-offs without QA gates.
version: "0.2"
kind: team-skill
roles:
  - id: operator-designer
    purpose: Ecosystem-first plan (cuBLAS/CUTLASS/cuDNN/CUB), then grid/tile/memory or library workspace; references vLLM, TensorRT-LLM, SGLang, FlashAttention, FlashInfer, Triton, DeepSeek official repos when relevant.
    skills: [cuda-kernel-design]
    tools: []
  - id: kernel-coder
    purpose: Implement per design: link CUTLASS/cuBLAS/etc., thin kernels, CMake/extension layout; mirror reference repos when specified.
    skills: [cuda-kernel-code-gen]
    tools: []
  - id: code-adversary
    purpose: Adversarial review for races, OOB, sync bugs, and numeric traps before validation.
    skills: [cuda-kernel-code-review]
    tools: []
  - id: precision-validator
    purpose: Golden-reference tests and numerical report (atol/rtol per dtype).
    skills: [cuda-kernel-precision-eval]
    tools: []
  - id: performance-optimizer
    purpose: Profile with Nsight/NCU, remove bottlenecks without breaking semantics.
    skills: [cuda-kernel-performance-optim]
    tools: []
---

# CUDA Operator Dev & Optimize Team

面向 **CUDA 自定义算子 / kernel** 的端到端流程：**5 个串行阶段 + Stage 3 对抗式代码审查门（C+A）**。CUDA 生态成熟，**默认「库与开源实现优先」**：在 **cuBLAS / cuBLASLt、CUTLASS、cuDNN、CUB（CCCL）** 等能覆盖的场景，设计应为 **封装与 launch/workspace**，而非从零写核心算子；**vLLM**、**TensorRT-LLM**、**SGLang**、**FlashAttention**、**FlashInfer**、**Triton / torch.compile（Inductor）**、**DeepSeek 官方开源**（如 DeepGEMM、DeepSeek-V3 等）等仓库中的组织方式与融合实现应作为 **对照示例**（见 [reference-ecosystem.md](reference-ecosystem.md)）。**默认优化的目标模型族为 Qwen 与 DeepSeek 推理栈**（算子形状、MoE、量化与验收默认见 [reference-target-models.md](reference-target-models.md)）；其他模型族须在 Pre-flight 中声明。确需手写 `__global__` 时，再在设计中展开 **grid/block/tile、shared/reg、occupancy**，并用 **golden + Nsight** 闭环。

## Workflow

0. **Pre-flight** — 阅读 [dependencies.yaml](dependencies.yaml)，向用户报告缺失项；子 skill 均可缺省（方法论在 `roles/*.md`）。阅读 [reference-ecosystem.md](reference-ecosystem.md) 与 [reference-target-models.md](reference-target-models.md)，确认库路径与 **目标模型族**（默认 Qwen / DeepSeek；否则由用户确认）。**环境**：`nvcc`/CUDA/驱动与 GPU；无 GPU 时 Final Report 标注 RUNTIME-DEGRADED。

1. **Design** — `operator-designer`：**先做生态策略**（库封装 vs CUTLASS vs 手写 kernel 及理由），再写 launch/tiling/memory；必选小节含 **Ecosystem Strategy** 与 **Target Model / Workload Alignment**。见 [workflow.md](workflow.md)。

2. **Code** — `kernel-coder`：按设计链接 **CUTLASS/cuBLAS/…** 或薄 kernel；CMake/FetchContent/子模块与参考仓库模式对齐。门禁：编译成功。

3. **Adversarial review** — `code-adversary`：独立挑错（OOB、竞态、`__syncthreads` 误用等）。门禁：LOW-RISK / ACCEPTABLE-RISK。

4. **Precision** — `precision-validator`：相对 golden（如 PyTorch CPU/CUDA ref）批量用例。门禁：PRECISION-PASS。

5. **Performance** — `performance-optimizer`：baseline profile → 优化 → 再 profile；报告 **Serving 对齐**（纯 kernel 与端到端可选）。语义不变；改 kernel 须再触发精度阶段。

6. **Final Report** — Leader 汇总，格式见 [workflow.md](workflow.md)。

## Roles

| id | Role file |
|----|-----------|
| operator-designer | [roles/operator-designer.md](roles/operator-designer.md) |
| kernel-coder | [roles/kernel-coder.md](roles/kernel-coder.md) |
| code-adversary | [roles/code-adversary.md](roles/code-adversary.md) |
| precision-validator | [roles/precision-validator.md](roles/precision-validator.md) |
| performance-optimizer | [roles/performance-optimizer.md](roles/performance-optimizer.md) |

派发子代理前：从对应 role 文件复制 **`## Inline Persona for Teammate`** 全文到 dispatch prompt。

## Files

| File | 用途 |
|------|------|
| [workflow.md](workflow.md) | Mermaid、步骤、门禁、最终报告模板 |
| [bind.md](bind.md) | 预算、行为约束、失败降级 |
| [dependencies.yaml](dependencies.yaml) | 工具与子 skill 清单 |
| [reference-ecosystem.md](reference-ecosystem.md) | 库优先级与 vLLM / TRT-LLM / SGLang / FlashAttention / FlashInfer / Triton 等参考示例 |
| [reference-target-models.md](reference-target-models.md) | Qwen / DeepSeek 算子关注点、MoE、量化与精度默认 |
| [README.md](README.md) | 项目定位与文档导航 |
