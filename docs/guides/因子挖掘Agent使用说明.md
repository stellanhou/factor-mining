# FM-v6 因子挖掘操作说明

因子挖掘输入研究问题、合同和逐小时币池，在 A 段探索并保留候选，冻结后执行一次 B 验证，最终写入合格创意卡。当前入口只支持 FM-v6。

## 准备输入

使用项目根目录 Python 3.12 环境，依赖安装见[项目说明](../../README.md)。行情数据库默认在 `market_data/crypto_quant.sqlite`。

| 输入 | 用法 |
| --- | --- |
| [正式合同](../../examples/factor_mining/research.contract.json) | 复制后填写新的 run_id 和研究问题；B 期限固定为 `[1, 4, 24]`，BH、方向 IC≥0.01、正价差 |
| [工程合同](../../examples/factor_mining/contract.example.json) | 配合[六资产币池](../../examples/factor_mining/universe.example.csv)验证流程；禁止入池 |
| 小时币池 CSV | 三列 timestamp、symbol、eligible；UTC 整小时、显式布尔值，覆盖每个小时和资产及预热 |
| 模型设置 | 默认使用项目配置的 MiMo；环境变量和可选 Codex 接口见项目说明 |

正式合同固定当前 A/B 区间，保留原数据的事后插值及既往使用说明。三轨只打标签，不能把硬门槛重新打开。旧 BY、baseline 和旧记录格式不能经当前执行入口恢复。

已有本地多年币池位于 `experiments/factor_mining/long_history_20260919_setup/universe.csv`；其日期目录是数据位置，不决定研究版本。重新准备输入可使用 `.venv/bin/python scripts/prepare_long_history_factor_mining.py prepare --contract <V6合同> --output-dir <新目录>`，默认输出目录为 `experiments/factor_mining/v6_inputs`。脚本直接读取 V6 合同，不依赖任何旧 Goal。输出目录必须尚不存在，成果目标另行填写；随后对同一输出目录执行 `verify`。

## 按步骤运行

以下 explore、validate 和补报告命令会使用所选真实模型。每次研究使用新的 run_id。

```bash
# A 段工程探索
.venv/bin/python -m crypto_quant.cli factor-mine explore \
  --contract examples/factor_mining/contract.example.json \
  --universe examples/factor_mining/universe.example.csv

# 从 a-complete.json 的 retained_ids 中选择实际候选；为空则不能冻结
.venv/bin/python -m crypto_quant.cli factor-mine freeze \
  --run-dir experiments/factor_mining/no_units_tags_example_20260928 \
  --candidate-ids candidate-0001 \
  --universe examples/factor_mining/universe.example.csv

# 对冻结候选执行一次 B 验证
.venv/bin/python -m crypto_quant.cli factor-mine validate \
  --run-dir experiments/factor_mining/no_units_tags_example_20260928 \
  --universe examples/factor_mining/universe.example.csv
```

计算、评估解释和优化决定分别保存。A 报告并列展示 1h、4h、24h 的共同样本结果。只有完整解释且明确 retain 的候选可以冻结；Optimizer 必须同时给出该候选的 `retained_horizons`。冻结后不能重开 A 或扩大期限。B 对冻结的每个期限分别评估，报告所有通过与未通过结果，并对全部“候选 × 期限”统一进行 BH 校正。

创意卡要求 A 保留期限与 B 通过期限有交集。卡片按候选生成，一项多期限研究仍只写一张卡；卡片保存逐期限摘要、覆盖、程序判定和归档引用。正式 B 门槛逐期限执行：校正通过、固定方向 IC 达到新合同的 `min_abs_ic`（研究示例为 0.01）、有向平均价差大于零；Plan3 轨道只作诊断标签。

2026-10-02 的 [IC 门槛调整](../plans/FM-v6_IC门槛调整Plan.md)使用全部 44 个原始 A 候选标定 0.02、0.015、0.01，选用 0.01 作为弱信号研究门槛。旧冻结合同及结果仍按原规则解释。复现标定与保存 B 数值对照：

```bash
PYTHONPATH=src .venv/bin/python scripts/calibrate_fm_v6_ic.py calibrate --output experiments/factor_mining/<new-calibration-id>
# 依据 A 证据记录 threshold_decision.json，并保存对应 research.contract.json 后执行：
PYTHONPATH=src .venv/bin/python scripts/calibrate_fm_v6_ic.py compare --comparison /Users/stellan/.codex/worktrees/e535/量化投资/experiments/factor_mining/horizon_comparison_20261001_v3/comparison.json --output experiments/factor_mining/<new-calibration-id>
```

该离线对照只统计数值合格身份，不写卡；新门槛在未参与选择的数据上的独立验证尚未完成。

## 历史研究工具

| 工具 | 固定范围与依赖 |
| --- | --- |
| [compare_fm_v6_horizons.py](../../scripts/compare_fm_v6_horizons.py) | 历史 Goal 的 26 个保留因子、7 个原批次，比较旧 24h 与多期限规则；依赖原 Goal、SQLite 归档、币池、冻结 B 面板、相应源码快照及原模型设置 |
| [calibrate_fm_v6_ic.py](../../scripts/calibrate_fm_v6_ic.py) | 原 Goal 全部 44 个 A 身份的 IC 标定及固定 26 个因子的保存 B 数值重判；直接依赖上方脚本的归档读取和校验函数，重判还依赖当前合同与准入实现 |

这两个入口是固定历史研究工具，主目录常规挖掘不依赖旧源码。多期限重放须显式提供已保存的旧 24h 源码路径，例如 [baseline_source/src](/Users/stellan/.codex/worktrees/e535/量化投资/experiments/factor_mining/horizon_comparison_20261001_v3/baseline_source/src)：使用 `--baseline-engine-root`，运行 old24 组时同时设置对应的 `--engine-root`。脚本原默认值指向当次临时源码目录，不应作为长期复现入口。

多期限工具的 `prepare` 会调用模型审阅 A 证据，`freeze-inputs` 会读取 B 行情，`run-group` 会生成隔离 Goal、报告和卡片；`recheck-completion` 会再次调用模型核验目标匹配。已有 [多期限对照证据](/Users/stellan/.codex/worktrees/e535/量化投资/experiments/factor_mining/horizon_comparison_20261001_v3/comparison.md)及 [IC 标定证据](/Users/stellan/.codex/worktrees/e535/量化投资/experiments/factor_mining/ic_calibration_20261002/)保留在 e535，本次主目录整合未复制实验产物或重跑研究。

## Goal 模式

Goal 由优化角色安排 A 研究任务，程序管理等待、恢复和计数。目标文件必须显式给出目标及成果数量，例如：

```json
{
  "goal_id": "my-factor-goal",
  "objective": "研究指定问题并形成通过 B 程序准入的因子创意卡",
  "target_ideas": 2
}
```

将目标与合同保存在自己的运行输入目录，并填写真实路径：

```bash
.venv/bin/python -m crypto_quant.cli factor-mine goal-start \
  --goal path/to/goal.json \
  --contract path/to/research-contract.json \
  --universe path/to/universe.csv

.venv/bin/python -m crypto_quant.cli factor-mine goal-status \
  --goal-dir experiments/factor_mining/goals/my-factor-goal

.venv/bin/python -m crypto_quant.cli factor-mine goal-resume \
  --goal-dir experiments/factor_mining/goals/my-factor-goal
```

`goal-status` 不调用模型或行情。`goal-resume` 复用保存的目标、数据和模型设置。已完成 Goal 不再新增研究；`Ctrl+C` 保存暂停。A 研究、成果核验与 B 解释使用独立上下文，B 结果不回流到下一轮研究选择。

Goal 的 cycle 采用当前引用格式，已有 cycle 复用原记录；不从旧版内嵌 JSON 恢复。目标只有在实际写卡、程序准入及目标匹配齐备后才计数。`waiting`、`paused`、`error` 和 `complete` 分别留存事件。

## 中断与补报告

B loader 调用之前即写访问标记。同一运行不能重新读取 B；已有完整数值检查点时，只补缺失解释：

```bash
.venv/bin/python -m crypto_quant.cli factor-mine complete-reports \
  --run-dir path/to/run --stage B
```

补报告不需要 db 或 universe。程序核对冻结证据和数值决定，复用已有解释与卡片。B 数值未完成的中断不能通过补报告绕过；需先检查原错误。A 补报告只能在冻结之前执行。

## 结果与检查

运行目录保存合同、候选过程、模型调用、报告和断点，逐期限数值证据写入 SQLite，创意卡默认位于 `experiments/idea_pool/`。B 报告与图表按实际冻结期限分别呈现；每张卡只对应一个候选，包含 A 保留期限、B 通过期限、完整程序判定和各期限归档引用。Goal 交付核验再次对照卡片来源、候选、公式、方向、期限、归档与程序判定；恢复时复用同一张卡和收据，不重复计数。供目标匹配使用的记录只包含 A 定义与证据，B 期限证据单独保存。这些本地产物不进入 Git。归档引用关系见[因子证据归档](因子证据归档.md)，实现职责见[模块说明](../../src/crypto_quant/research/factor_mining/README.md)。

```bash
.venv/bin/python -m pytest -q tests
```

测试用本地模型替身和临时数据验证流程。真实研究的程序版本、数据用途、运行进度和统计结果分别记录。

当前实现的测试位于 `tests/`。`scratch/retired_factor_v1_v5_20260930/` 保存退役版本材料，其旧测试不纳入当前 FM-v6 回归。
