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

新运行同时展示三个观测间隔 Δ=1h、4h、24h 的排名变化 D，包括覆盖、缺失及阶段差异。D 越低表示共同币种的相对排序越稳定，不等于实际成交费用越低。预测期限 H 与 Δ 分开使用；所有 Δ 均保留，不自动选择最小值作为总分。旧记录缺少该诊断时明确显示“未计算”。

Optimizer 可提出 `rank_displacement` 受控修改：在生成变体公式前明确 H、Δ、D 的绝对改善下限与允许的方向 IC 损失。新旧因子必须在两端及标签均有效的共同样本上配对比较，程序根据区间决定继续、停止或暂停；停止一条修改路线不会自动淘汰原候选。该能力不默认给所有公式平滑，不改变 B 准入，也不自动恢复已暂停的 Goal。完整定义见[模块说明](../../src/crypto_quant/research/factor_mining/README.md#信号持续性与受控修改)。

创意卡要求 A 保留期限与 B 通过期限有交集。卡片按候选生成，一项多期限研究仍只写一张卡；卡片保存逐期限摘要、覆盖、程序判定和归档引用。正式 B 门槛逐期限执行：校正通过、固定方向 IC 达到新合同的 `min_abs_ic`（研究示例为 0.01）、有向平均价差大于零；Plan3 轨道只作诊断标签。

2026-10-02 的 [IC 门槛调整](../plans/FM-v6_IC门槛调整Plan.md)使用全部 44 个原始 A 候选标定 0.02、0.015、0.01，选用 0.01 作为弱信号研究门槛。旧冻结合同及结果仍按原规则解释。复现标定与保存 B 数值对照：

```bash
PYTHONPATH=src .venv/bin/python scripts/calibrate_fm_v6_ic.py calibrate --output experiments/factor_mining/<new-calibration-id>
# 依据 A 证据记录 threshold_decision.json，并保存对应 research.contract.json 后执行：
PYTHONPATH=src .venv/bin/python scripts/calibrate_fm_v6_ic.py compare --comparison /Users/stellan/.codex/worktrees/e535/量化投资/experiments/factor_mining/horizon_comparison_20261001_v3/comparison.json --output experiments/factor_mining/<new-calibration-id>
```

该离线对照只统计数值合格身份，不写卡；新门槛在未参与选择的数据上的独立验证尚未完成。

## 构想时查看因子树

新的构想轮次会自动收到 A 段因子图谱，无需增加命令参数。成员来自当前运行此前的 A 研究和该 Goal 已传入的 A 历史，包含未成功的尝试；不会自动扫描正式卡片池或所有历史目录。首轮没有历史时显示空图谱。

图谱用当前 A 面板重算已执行公式的信号相关性，提供完整层次树、分支成员、代表公式、字段覆盖和历史 A 证据。数量较多只说明该分支已有较多研究，不能据此禁止继续构想；字段使用较少也不等于存在有效信号。无法可靠计算相关性的候选保留状态和原因，不会被误标为新方向。

每轮快照保存在 A 记录中，并在模型请求里引用。恢复未完成的构想请求会复用对应快照；原研究记录仍可回查。该改造帮助识别重复与提出差异假设，是否提高有效因子产出仍需相同预算下的研究对照。计算方法及数据边界见[模块说明](../../src/crypto_quant/research/factor_mining/README.md#hac-因子研究上下文)。

## 历史研究工具

| 工具 | 固定范围与依赖 |
| --- | --- |
| [compare_fm_v6_horizons.py](../../scripts/compare_fm_v6_horizons.py) | 历史 Goal 的 26 个保留因子、7 个原批次，比较旧 24h 与多期限规则；依赖原 Goal、SQLite 归档、币池、冻结 B 面板、相应源码快照及原模型设置 |
| [calibrate_fm_v6_ic.py](../../scripts/calibrate_fm_v6_ic.py) | 原 Goal 全部 44 个 A 身份的 IC 标定及固定 26 个因子的保存 B 数值重判；直接依赖上方脚本的归档读取和校验函数，重判还依赖当前合同与准入实现 |

这两个入口是固定历史研究工具，主目录常规挖掘不依赖旧源码。多期限重放须显式提供已保存的旧 24h 源码路径，例如 [baseline_source/src](/Users/stellan/.codex/worktrees/e535/量化投资/experiments/factor_mining/horizon_comparison_20261001_v3/baseline_source/src)：使用 `--baseline-engine-root`，运行 old24 组时同时设置对应的 `--engine-root`。脚本原默认值指向当次临时源码目录，不应作为长期复现入口。

使用当前引擎导入固定历史 A 证据时，工具从原评估对应的归档因子值及同源 A 币池计算排名变化，写入新运行的诊断归档。归档 CSV 的数值列使用 `float_precision="round_trip"` 读取，保留原浮点值的区别，避免解析产生额外并列排名。缺少原值集时直接报错；不改写源评价，也不换库补齐。旧源码引擎仍使用其原评价定义。

多期限工具的 `prepare` 会调用模型审阅 A 证据，`freeze-inputs` 会读取 B 行情，`run-group` 会生成隔离 Goal、报告和卡片；`recheck-completion` 会再次调用模型核验目标匹配。已有 [多期限对照证据](/Users/stellan/.codex/worktrees/e535/量化投资/experiments/factor_mining/horizon_comparison_20261001_v3/comparison.md)及 [IC 标定证据](/Users/stellan/.codex/worktrees/e535/量化投资/experiments/factor_mining/ic_calibration_20261002/)保留在 e535，本次主目录整合未复制实验产物或重跑研究。

## Goal 模式

Goal 由优化角色安排 A 研究任务，程序管理等待、恢复和验收。数量目标显式给出 `target_ideas`，已有目标继续沿用原计数规则，例如：

```json
{
  "goal_id": "my-factor-goal",
  "objective": "研究指定问题并形成通过 B 程序准入的因子创意卡",
  "target_ideas": 2
}
```

本轮[质量目标](../../examples/factor_mining/quality.goal.json)使用 `quality_target`，与 `target_ideas` 二选一：预测期限 H=4h、排名变化间隔 Δ=4h，配对 D 改善的区间下界至少 0.01，方向 IC 变化的区间下界不低于 −0.001。区间置信度来自研究合同，正式合同为 95%。这些数值是本轮事先声明的验收要求，没有从历史实验中确定为最优门槛。

只有候选完成预声明的 A 段受控修改、程序配对判定 `continue`、满足上述区间要求、通过同一 4h 期限的原有 B 准入，并经目标内容核验，质量 Goal 才完成。普通 B 合格卡和模型肯定意见不能单独完成质量目标。图谱用于支持差异化构想，实际组合增量仍留待后续检验。B 的 BH、IC 下限和正价差规则不变，B 的 D 继续只作诊断。

质量目标的 `max_cycles` 可以省略或设为 `null`，表示持续研究直到质量目标达成；正整数仍表示可选的批次上限。有限预算用尽后状态为 `budget_exhausted`，不算完成。取消已有上限时，在运行进程退出后追加 Goal 状态 `max_cycles: null`，保留原 `goal.json`；新进程的 `goal-resume` 从下一轮接续，不重跑已完成批次，也不重读其 B 数据。已完成 Goal 始终不再新增研究。

四角色模型可用 `--role-models` 指定。[本轮配置](../../examples/factor_mining/role-models.json)为构想 Astra/max、计算 Luna/max、评估 6.1 Sol/high/Fast、优化 6.1 Sol/max/Fast。Sol 两项的 `service_tier="priority"` 是当前 Codex 模型目录中的 Fast 档位标识，与推理强度分开设置。请求显式指定档位并核对运行时接受结果；不支持或未接受 Fast 时直接停止，不自动降档。Fast 设置仅作用于这两个角色的研究进程，保存在 Goal 模型设置及调用留档中。计算角色的模型仅负责报错后的公式修复；因子值和数值检验由程序计算。四角色必须全部配置；纠正和证据补读继续使用原角色的模型，不自动切换。Goal 保存每个模型的完整接口及运行时设置，恢复时逐项核验。

本轮启动方式如下；先复制研究合同并填写新的 `run_id`，币池沿用已核验的输入。当前项目随 SDK 的旧运行时尚不列出完整 GPT-6 模型，因此通过 `--codex-bin` 显式使用已核验的桌面运行时，不修改全局配置：

```bash
.venv/bin/python -m crypto_quant.cli factor-mine goal-start \
  --goal examples/factor_mining/quality.goal.json \
  --contract path/to/new-research-contract.json \
  --universe experiments/factor_mining/long_history_20260919_setup/universe.csv \
  --provider codex \
  --role-models examples/factor_mining/role-models.json \
  --codex-bin "$(command -v codex)"
```

使用角色配置时不再指定共享 `--model` 或 `--reasoning-effort`。单模型入口仍使用原参数。

2026-10-04 本轮扩展为六组并发：H=1h、4h、24h，每个期限分别使用混合模型与全 Luna；各组的排名变化间隔 Δ 与 H 一致。全 Luna 组仅替换模型名，保留相同角色的推理强度及评估、优化角色的 Fast 档位。六组使用相同的 A/B 数据、币池和质量门槛，独立保存研究历史、归档及卡片。初始每组 12 批上限已按用户要求取消，后续持续到质量 Goal 达成。配置、原始源码快照、数据来源与比较范围保存于[本轮清单](/Users/stellan/量化投资/experiments/factor_mining/quality_comparison_20261004/six-arms-125251/manifest.json)；预算取消与安全接续的实际生效记录保存于恢复清单、追加 Goal 状态及 `monitoring-handoff.json`。旧进程保留运行，在原上限自然退出后由半小时巡检接续。配置检查在真实模型启动前完成；当前 A 输入另行预检，B 行情在各组冻结后才读取。原 `quality.goal.json` 为 4h 单组无上限示例，六组的初始输入分别位于本轮各组目录的 `goal.input.json`。

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

Goal 的 cycle 采用当前引用格式，已有 cycle 复用原记录；不从旧版内嵌 JSON 恢复。数量目标只有在实际写卡、程序准入及目标匹配齐备后才计数；质量目标另外保存 A 配对证据引用、目标及区间结果。`waiting`、`paused`、`error`、`budget_exhausted` 和 `complete` 分别留存事件。

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
