# Execution Guardrails (CUDA)

## Resource Constraints

| Item | Limit | Reason |
|------|-------|--------|
| max_parallel_teammates | 1 | 串行 C-pattern |
| total_wall_clock_budget | 45 min | 含 kick-back |
| total_token_budget | 300k | 跨 5 角色 |
| operator-designer_token_budget | 40k | |
| kernel-coder_token_budget | 60k | 代码体积大 |
| code-adversary_token_budget | 50k | |
| precision-validator_token_budget | 60k | |
| performance-optimizer_token_budget | 80k | 多轮 profile |
| per-role wall_clock | 5–10 min | 与整包 45min 总预算衔接 |

## Behavioral Constraints

- **Leader 只编排**：不写 kernel、不代替审查/测试/profile。
- **阶段边界**：上游产出不可被下游改写语义；`kernel-coder` 不得重画 tiling 蓝图；`performance-optimizer` 不得改数学定义（仅等价变换或实现层优化）。
- **对抗隔离**：`code-adversary` 仅见最终代码 + 设计文档，不见 coder 自述。
- **Kick-back**：失败只回传必要条目，不回传下游全文。
- **精度优先**：优化若改变数值，须重新走 `precision-validator`。
- **真机 GPU**：执行或 profile 前须用户确认环境与驱动；无 GPU 时在 Final Report 标注 DEGRADED。

## Failure Handling

- Teammate 超时：重试 1 次；再失败标记 `[ROLE MISSING — timeout]`。
- 输出不符合 Output Schema：重试 1 次并内联 schema。
- Adversary 敷衍（无 ≥3 类检查 pass）：按 Mandatory 重派。
- 同 gate 连续 3 次失败：**BLOCKED**，交由用户决策。

## Degraded Mode

| 条件 | 行为 |
|------|------|
| 单次需求含 >5 类无关算子 | 收敛为单一算子并警告 |
| 设计文档极长 | 设计阶段可摘要，下游标注 CONTEXT-RISK |
| 源码 >2000 LOC | Adversary 改为 Top-3 高风险区深度审查 |
| 无 GPU / 无 Nsight | Performance 阶段仅代码审查级建议 + 理论瓶颈 |
