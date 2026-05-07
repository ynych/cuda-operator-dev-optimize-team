---
name: cuda-operator-dev-optimize-team
description: |
  Team skill: CUDA custom operators and kernels — design, implement, adversarial review, golden precision, Nsight/NCU performance. Prefer CUTLASS, cuBLAS, cuDNN, CUB before raw __global__.
  TRIGGER when user: writes or optimizes CUDA / CUDA kernel / custom op / extension / .cu; matrix transpose, GEMM, reduction, elementwise, fusion; gives shapes (M=, N=, batch) only; points to solution.cu or kernel path; provides Python/C++ reference for golden test; asks for nvcc build, profile, NCU, occupancy, coalescing.
  触发词（中英）: CUDA、kernel、算子、矩阵转置、优化、手写、从零、solution.cu、ref、golden、精度、profile.
  LLM stack references when relevant: vLLM, TensorRT-LLM, SGLang, FlashAttention, FlashInfer, Triton, Transformer Engine, quantization, DeepSeek, Qwen (see reference-ecosystem.md, reference-target-models.md).
  Non-LLM kernels (e.g. transpose): use Target Model / Workload Alignment = generic/other in Stage 1; full gates still apply unless user explicitly opts out per bind.md.
version: "0.2.1"
kind: team-skill
roles:
  - id: operator-designer
    purpose: Ecosystem-first plan (cuBLAS/CUTLASS/cuDNN/CUB), then grid/tile/memory or library workspace; references vLLM, TRT-LLM, SGLang, FlashAttention, FlashInfer, Triton, TE, Apex, quantization stacks, CUDA Graph, DeepSeek official repos when relevant.
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

## 如何触发（编排 / 用户话术）

自动匹配主要依赖本文件 **YAML `description`** 中的英文/中文关键词。若未自动挂上本 skill，请用户 **显式点名** 或复制下面任一句（可改路径与 shape）：

| 场景 | 示例（可直接改参数） |
|------|----------------------|
| 从零只有需求 | 「用 **cuda-operator-dev-optimize-team** 全流程：帮我写一个 **矩阵转置的 CUDA kernel** 并 **优化**。」 |
| 只有 shape / 算子名 | 「用团队 skill **优化 matrix_transpose**，**M=6000，N=7000**。」 |
| 已有 `.cu` | 「用 **cuda-operator-dev-optimize-team** **优化** `solution.cu` 里的 **matrix_transpose**，**M=6000，N=7000**。」 |
| 实现 + reference | 「用团队 skill：**优化** `kernel/matrix_transpose/solution.cu`，**ref** 是 `kernel/matrix_transpose/transpose_ref.py`，**M=6000，N=7000**。」 |

**Leader 注意**：非 Qwen/DeepSeek 类算子（如通用转置）在 Stage 1 的 **Target Model / Workload Alignment** 中填 **generic / other**，并简述 shape、dtype、无 MoE；其余门禁不变。

---

面向 **CUDA 自定义算子 / kernel** 的端到端流程：**5 个串行阶段 + Stage 3 对抗式代码审查门（C+A）**。CUDA 生态成熟，**默认「库与开源实现优先」**：在 **cuBLAS / cuBLASLt、CUTLASS、cuDNN、CUB（CCCL）** 等能覆盖的场景，设计应为 **封装与 launch/workspace**，而非从零写核心算子；**vLLM**、**TensorRT-LLM**、**SGLang**、**FlashAttention**、**FlashInfer**、**Triton / torch.compile（Inductor）**、**Transformer Engine**、**Apex**、**llm-compressor / AWQ / GPTQ / SmoothQuant**、**CUDA Graph** 与 **DeepSeek 官方开源**（如 DeepGEMM、DeepSeek-V3 等）等仓库或文档中的组织方式与融合实现应作为 **对照示例**（见 [reference-ecosystem.md](reference-ecosystem.md)）。**默认优化的目标模型族为 Qwen 与 DeepSeek 推理栈**（算子形状、MoE、量化与验收默认见 [reference-target-models.md](reference-target-models.md)）；其他模型族须在 Pre-flight 中声明。确需手写 `__global__` 时，再在设计中展开 **grid/block/tile、shared/reg、occupancy**，并用 **golden + Nsight** 闭环。

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
| [reference-ecosystem.md](reference-ecosystem.md) | 库优先级与推理栈、TE/Apex、量化工具链、CUDA Graph、FlashInfer 等参考示例 |
| [reference-target-models.md](reference-target-models.md) | Qwen / DeepSeek 算子关注点、MoE、量化与精度默认 |
| [README.md](README.md) | 项目定位与文档导航 |
