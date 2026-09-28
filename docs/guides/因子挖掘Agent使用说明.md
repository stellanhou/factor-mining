# 因子挖掘 Agent 使用说明

更新：2026-09-15。

因子挖掘已接入「量化投资」主项目，作为创意库的一个来源。代码位于 [research/factor_mining](../../src/crypto_quant/research/factor_mining)，复用已有本地行情、35个字段和算子接口。

## 当前能做什么

1. 构想 Agent 输出公式、具体计算含义和待验证的金融解释。
2. 计算 Agent 直接执行合法公式；公式报错时依据程序错误尝试修复。候选含义和假设保留在研究记录中，不再进行六维含义审核或按解释文字退回。
3. 程序计算未来24小时的 Rank IC、分组收益、稳定性及统计证据，评估 Agent 只解释完整结果、证据限制和研究建议，不输出候选去向或创意卡准入决定。
4. 优化 Agent 逐个候选回答四个研究问题，决定继续优化、保留待验证、暂停或淘汰；获准优化时只提出修改任务和配对检验合同，写明原候选、依据、核心假设、已观察问题、语义改动目标、固定部分、对照指标及停止／暂停条件。它不输出新候选定义或最终公式；下一轮构想 Agent 首次生成完整候选。第四问只判断继续是否带来新的可验证信息，不判断剩余预算。支持保留当前版本的同时研究改进版。
5. A段探索结束后冻结候选与B段历史币池，再完成一次B段批量验证。程序先按冻结规则记录通过情况和交付资格，评估 Agent 再解释结果；符合条件且解释报告完整的候选生成创意卡。C段留给后续完整策略。

研究记录保存在 `experiments/factor_mining/<运行ID>/`，创意卡保存在 `experiments/idea_pool/`。创意卡保留主 Plan 要求的8项内容，状态为 `research_idea`。

评估报告现有6项核心内容：分析、经济机制、适用条件、证伪条件、证据限制、下一步研究建议，已移除旧的`decision`。

### 四个Agent统一的格式接收规则

构想、计算、评估、优化四个角色，以及外层消息和原文读取请求，统一在模型网关检查：必需字段和原生类型保留，额外字段记录后排除，不参与后续执行。递归处理候选定义、计算检查项、修改任务、配对检验合同、建议处置、四问及去向等嵌套对象。Optimizer旧`candidate`字段如果仍被模型输出，只作格式偏差留痕，不会交给Ideator。

原始回复完整保存在`model_calls/`；偏差记录位于A或B记录目录的`model-<请求编号>-format.json`，包含角色、额外字段的完整JSON路径和原文位置。A／B报告展示格式记录。向下一环节或原文读取后的继续请求传递时，仅使用声明字段，附加字段值不混入研究记录。

缺失必需字段、类型错误、非法取值、证据错配、含义不一致等仍按原规则检查；不把字符串布尔值自动转换为布尔值，也不补造缺失内容。此规则针对模型回复，不放宽研究合同、数据接口或冻结证据的校验。

### API失败重试

当前示例合同已设`output_tokens=null`，不再向接口发送人为指定的单次输出上限（原值16384）。接口按自身默认设置生成，实际用量照常保存。本地上下文容量仍单独管理；总调用次数上限已取消。

- 首次请求失败后最多重试5次，合计至多6次尝试；累计调用次数不再导致提前停止，每次请求仍单独留档。
- 空正文、连接失败、超时、HTTP 408／429／5xx及服务返回结构异常会重试。鉴权／参数错误、内容过滤仍明确停止；模型正文格式或研究条件错误进入下述回复纠正流程。
- 各次重试等待1、2、4、8、16秒，重发相同请求，接着当前Agent继续；前面的研究步骤与B数据加载不会重新执行。
- 每次请求／回复分别保存，空回复也保留原始返回、结束原因及用量；错误记录注明尝试次数、是否重试和停止原因。原始请求不记录鉴权头，返回中已知密钥脱敏。

### 回复纠正与单候选故障隔离（2026-09-15）

- 每次角色任务收到无效回复后，按当前运行契约给模型纠正机会。覆盖非空截断、JSON格式、缺字段、类型、错误证据引用及研究约束错误；解释文字不再因语义判断被退回。
- 纠正失败的计算核对记录为当前候选的`model_review_failed`，其他候选继续。解释失败记为`report_pending`，已完成的数值仍保留；优化Agent能看到该缺口，不把解释缺失当作因子无效。
- 构想／优化整体回复仍无法纠正时停止，错误回复不会占用候选、建议或路线状态。鉴权错误、证据被改动、数据边界错误及内部程序错误仍明确停止。
- 已取消`max_model_calls`总调用次数限制，也取消A段`rounds`、`max_candidates`、`candidates_per_round`和`max_route_attempts`。API重试、回复纠正、原文读取和报告补全不因累计次数停止；上下文容量仍作为模型接口技术边界单独管理。

### A段自然终止

A段不设置轮数、候选总数、单轮候选数或路线尝试额度。每轮实际数量和路线尝试次数仍保存在研究记录中；优化Agent没有获准修改建议、返回空`proposals`时，循环自然结束。程序不再向优化Agent发送`remaining_budget`，修改设计也不再包含`attempt_budget`。API重试与回复纠正上限用于处理单次故障，不属于研究循环额度。

### 补齐已计算候选的解释

B段先保存整批数值、统计校正及通过／未通过判定，再逐个生成解释。缺少解释时，命令返回`status=reports_pending`和`pending_report_ids`，`B-report.md`也列出缺口。`b-numerical-complete.json`标记整批数值已完成；补解释时程序重新核对批次校正与准入决定，全部解释完成后才写最终`validation.json`。满足程序资格且解释完整的候选可交付创意卡。

```bash
# 补齐A段解释：必须在冻结B之前；不重开构想／优化
.venv/bin/python -m crypto_quant.cli factor-mine complete-reports \
  --run-dir experiments/factor_mining/<运行ID> --stage A \
  --model gpt-5.6-luna --reasoning-effort max

# 补齐B段解释：只读取完整数值快照，不重读B行情、不重算因子
.venv/bin/python -m crypto_quant.cli factor-mine complete-reports \
  --run-dir experiments/factor_mining/<运行ID> --stage B \
  --model gpt-5.6-luna --reasoning-effort max
```

补全入口无需行情数据库或币池参数。已有解释会跳过，已有创意卡经一致性核对后复用；原始记录和失败历史不改写，Markdown报告根据现有证据重新生成。A段冻结仍要求所选保留候选的解释完整；`a-complete.json.pending_report_ids`保留探索结束时快照，补全后的状态见命令返回和A报告。

新入口保留合同和冻结证据检查。旧合同用于新运行时需移除`max_model_calls`、`allowed_changes`、`rounds`、`max_candidates`、`candidates_per_round`和`max_route_attempts`字段，优化建议中的`design.change_kind`和`design.attempt_budget`也不再使用。B数值尚未整批完成的运行也不能只靠补解释完成验证。

### B验证与创意卡

B段数值规则继续检查批次校正、固定方向的IC、组间收益差和阶段重复。程序在`b_records/<候选ID>-validation.json`保存`validation_status`（`passed`／`not_passed`）、`decision_source=program`、`eligible_for_idea_pool`、原因及检验结果；评估 Agent 收到这份记录后作解释。工程样例即使数值满足要求，交付资格仍为false。

创意卡保留主Plan八项内容，关联A段优化Agent的保留决定和B段程序准入证据，新增`b_validation_status`；完整策略状态仍为`not_started`。B段未通过会保留原因和报告，不会自动反馈给A循环修改公式，也不自动转用C。

### 查看候选去向

- `A-report.md` 的“候选去向”展示每个版本的最终决定、四个问题的回答、证据链接及暂停恢复条件。
- `A-report.md` 的“修改任务与配对检验合同”展示Optimizer声明的证据、改动边界和预定判断规则；它们不包含最终公式，也不能代替配对结果。
- `a_records/round-XXX-optimization.json` 保留每一轮的完整决策及`proposal.modification_task`/`proposal.experiment_design`；下一轮Ideator实际收到这些任务、配对合同和当前决定。
- `a-complete.json` 的 `retained_ids` 是可选的B冻结名单；`evaluated_ids` 仅表示做过评估，不能作为准入名单。
- 普通`explore`在无获准优化建议时结束A循环。Goal模式会交回优化Agent安排下一方向，或等待明确的数据依赖；候选暂停不会直接暂停整个Goal。
- 研究理由由模型判断，程序检查引用、类型、状态与检验设计；2026-09-15真实试跑在评估回复格式检查处停止，尚未到达四问决策，详见下方。

### 假设指导的修改边界

相同核心假设下，优化Agent可在`change_target`中指定多个相互关联的语义改动，但不输出可执行公式。构想Agent只能采用或放弃任务；采用时生成完整候选，并在`change_reason`说明公式如何符合核心假设、改动目标和固定部分。它不能改写Optimizer预先声明的配对判断标准。程序继续限制公式白名单、因果数据边界、父版本与建议关联、固定方向及同网格配对比较；文本完整只说明任务已写清，实际配对结果才判断修改是否有帮助。

## 已完成的验证

### 2026-09-15 Optimizer与Ideator职责拆分

Optimizer的`proposal`已删除完整`candidate`和`formula_alignment`，改为`modification_task`与`experiment_design`。下一轮Ideator只能采用或放弃任务；采用时首次生成完整候选和可执行公式。程序校验任务证据必须真实存在且包含对照候选的评估记录，再核对父版本、proposal、固定方向和配对合同。

全量104项测试通过。新测试确认Optimizer记录和下一轮上下文不包含公式，Ideator能根据任务生成并关联新候选；模型若仍输出旧`candidate`字段，其原文只作格式偏差留痕，字段和公式不进入研究状态。伪造证据引用及未引用对照评估的任务均被拒绝。本次未调用真实模型。

### 2026-09-15 A段改为自然循环

研究合同和流程已删除轮数、候选总数、单轮候选数及路线尝试额度。全量103项测试通过；其中新增场景连续运行6轮、累计7个候选，超过旧示例边界后仍继续，最终仅因优化Agent返回空`proposals`结束。测试确认优化请求不再包含`remaining_budget`，修改设计不再包含`attempt_budget`。本次只完成代码与离线验证，尚未用真实模型验收自然循环。

### 2026-09-15 四个Agent统一格式处理

完整测试86项通过，覆盖四个角色与嵌套对象、外层消息和原文读取请求。上次多出内层`read_records`的真实构想回复已通过离线回放，两个候选的核心内容保持原值；本次没有新发真实模型请求，旧运行仍保持停止状态。[核对记录](../../experiments/factor_mining/unified_format_verification_20260915.json)。

### 2026-09-15 取消单次输出上限后的试跑

示例合同已取消16384单次输出限制，彼时83项测试通过。接着的1次真实调用完整返回2个候选（`finish_reason=stop`），但构想结果额外返回内层`read_records`，触发当时的构想格式检查并停止。该次没有截断；当时附加字段处理只覆盖评估角色，现已按上方统一规则改造。连同此前截断运行共3次调用，原24次总预算未超出。

- [最新运行结果](../../experiments/factor_mining/go_deepseek_default_output_20260915_052258/运行结果.md)

### 2026-09-15 带重试机制真实试跑：计算回复被截断

沿用原六币四天样例实跑约1分49秒、2次请求。构想Agent提出2个候选，计算Agent核对第一个候选时达到16384 token输出上限，其中15281用于推理；服务返回`finish_reason=length`，正文与JSON均未完成，程序停止。原始回复和用量本次完整保留。当前重试条件不含“非空但被截断”，实际重试0次；评估、四问和最终去向均未运行。

- [最新测试结果](../../experiments/factor_mining/go_deepseek_retry_20260915_051451/运行结果.md)
- [输出长度与用量核对](../../experiments/factor_mining/go_deepseek_retry_20260915_051451/real-model-verification.json)

此结果确认本次截断来自输出上限；此前空回复的具体原因仍未确认。本轮没有改代码、提高预算或手动收尾，也没有读取B/C。

### 2026-09-15 API重试机制

完整测试82项通过，新增覆盖首次空回复后恢复、连续5次重试后成功、重试上限、总预算提前耗尽、鉴权／参数错误不重试、两种接口保留空回复原始诊断，以及重试不重跑A研究或B数据加载。所有网络故障均为测试模拟，本次没有新发真实模型请求。

### 2026-09-15 评估职责与B准入分离

已删除评估角色的旧去向字段；A段四问决定继续由优化Agent负责，B准入取消评估Agent的“必须keep”条件。测试分别让模型额外输出旧`keep`与`discard`，确认这两个值都不能改变程序的B判定；原有统计、方向、幅度、阶段和工程用途限制继续生效。此改动没有调用真实模型，优化Agent空回复问题尚未处理。

完整测试76项通过，历史原始回复的离线回放也确认旧`decision`只作附加字段留痕。[核对记录](../../experiments/factor_mining/evaluator_role_review_20260915.json)。

### 2026-09-15 再次真实试跑：优化回复为空

修正评估附加字段处理后，沿用相同六币、四天、DeepSeek V4.1 Flash（low）再跑。约3分45秒、6次调用，两份评估均正常接收；第6次优化Agent调用正文为空而停止，四问及最终去向没有生成。两个候选为永续—现货溢价拥挤度，以及资金费率×持仓增长拥挤度。

本轮未出现附加字段，未触发其处理分支；该分支的验证来自下方离线回放。优化请求确实包含两个候选的完整评估及报告。空回复的原始返回体和usage未保存，具体原因尚未确认。没有重试、手动收尾、B/C读取或创意卡输出。

- [最新真实运行结果](../../experiments/factor_mining/go_deepseek_fields_20260915_042406/运行结果.md)
- [最新运行核对](../../experiments/factor_mining/go_deepseek_fields_20260915_042406/real-model-verification.json)

### 2026-09-15 评估附加字段处理

上次多出`conditions_note`的原始回复已通过离线回放：7个核心字段保持原值并通过校验，额外字段不进入报告数据，原始记录未改动。本次没有调用真实模型或恢复旧运行；评估内容的正确性仍需单独处理。

完整测试75项通过，包含A／B附加字段处理、原文保留，以及缺失字段、类型错误和非法取值仍被拒绝。

- [原始失败回复的离线回放记录](../../experiments/factor_mining/report_format_replay_20260915.json)

### 2026-09-15 新流程真实试跑：未完成

按用户要求，用6币、2026年7月1日至4日A段接入DeepSeek V4.1 Flash（low）。实际调用5次，两个候选完成公式检查与数值评估；第二份模型评估多返回`conditions_note`，触发格式检查并停止。未调用优化Agent，未产生四问回答或最终去向，未读取B/C，没有重试、手动收尾或创意卡输出。

复核另发现：第一份评估误读已经做过方向调整的收益差符号；两份评估倾向把弱证据直接作为淘汰理由。当前只能确认构想和数值计算可运行，不能把此次试跑算作新决策流程的成功验收。

- [查看实际因子、数值及问题](../../experiments/factor_mining/go_deepseek_decisions_20260915_0406/运行结果.md)
- [真实运行与用量核对](../../experiments/factor_mining/go_deepseek_decisions_20260915_0406/real-model-verification.json)

### 2026-09-15 模拟模型验证

2026-09-15的新决策流程：**72项测试通过**，包括四种去向的实际分流、保留原版并优化、停止修改路线但保留原版、无建议提前结束、错误引用／矛盾回答拒绝、预算检查和禁止自动恢复暂停候选。

本地六币两轮工程联调使用真实行情和模拟模型回复：第一轮4个候选覆盖4种去向，第二轮仅为获准优化的候选创建第5个版本；暂停和淘汰版本无法冻结。3个最终保留版本完成工程B验证，未生成创意卡，原始数据库未变、C未读取，共17次模拟请求、0次真实模型调用。此记录验证程序和上下文衔接，真实模型的研究判断质量尚未验收。

- [新决策流程A段报告](../../experiments/factor_mining/decision_routing_local_20260915/A-report.md)
- [新决策流程核对结果](../../experiments/factor_mining/decision_routing_local_20260915/engineering-verification.json)

以下为此前版本的真实模型联调：

2026-09-14已使用 **DeepSeek V4.1 Flash** 完成六币、两轮真实联调，4条候选完成计算与评估，模型结论为2条建议淘汰、2条证据不足。第二轮实际收到前一轮的完整逐期结果、模型报告与优化诊断。

联调修正了输出字段／类型说明，并调整了推理与上下文预算。最后一次优化返回空正文后，使用剩余的一次请求明确完成收尾评审；包括前期诊断尝试，共发起24次模型请求。运行记录标明这次手动收尾，没有把它记成一次完全无人干预的成功执行。

- [真实模型运行报告](../../experiments/factor_mining/go_deepseek_v41_flash_20260914_low/A-report.md)
- [真实联调核对结果](../../experiments/factor_mining/go_deepseek_v41_flash_20260914_low/real-model-verification.json)
- [收尾恢复记录](../../experiments/factor_mining/go_deepseek_v41_flash_20260914_low/finalization-recovery.json)

以下为此前的离线验证记录：

历史联调记录（取消含义审核之前）：本地六币样本曾跑通两轮，真实行情生成612行×35字段的A段面板，3条候选包含一次修复、一次定义不足退回，以及下一轮的窗口修改检验；随后完成冻结批次的B段验证。原始数据库未修改，未读取C段。

这次离线联调用脚本模拟模型回复，检查程序与上下文衔接。上方新增真实模型联调也只使用工程样例，未开展正式B段验证，没有把候选加入创意库。

- [联调核对结果](../../experiments/factor_mining/engineering_local_20260914_final/engineering-verification.json)
- [A段报告及分组收益图](../../experiments/factor_mining/engineering_local_20260914_final/A-report.md)
- [B段报告及分组收益图](../../experiments/factor_mining/engineering_local_20260914_final/B-report.md)

## 接入官方 Codex Python SDK

项目统一使用 Python 3.12.13 和根目录 `.venv`。安装完整环境：

```bash
uv sync --frozen --all-extras
codex login
.venv/bin/python -m crypto_quant.cli factor-mine models
```

当前默认 `gpt-5.6-luna`、`max` 推理强度，通过官方 `openai-codex==0.154.0` 及其锁定的本地运行时调用，使用 ChatGPT 订阅登录。无需 OpenCode 密钥、CPA 或单独模型服务；不自动回退到 API 计费或其他模型。可用模型以当前账号的 `models` 返回为准。

因子挖掘、Goal 优化角色、策略研究、扫描解读和历史候选重评都使用同一适配器。每次调用建立独立临时会话，关闭研究外工具、继承的 MCP/插件和个人记忆；原始消息角色及正文完整传入，模型流事件与用量保存在研究记录中。SDK 本身的基础上下文仍占用订阅额度。单次默认等待300秒，可通过 `--timeout-seconds` 调整；超时、额度不足或认证错误停止并保留检查点，不偷偷更换线路。

SDK 不提供单次输出 token 上限；新合同须使用 `output_tokens=null`，数值上限会在调用前被拒绝。模型的实际上下文容量另由服务决定。旧 Go 运行的原始记录与报告保留；能否续跑由现有合同、输入和证据检查决定。

环境与安装详见 [项目说明](../../README.md)，接口来源见 [官方 SDK 文档](https://learn.chatgpt.com/docs/codex-sdk)。

## 运行前准备

- **研究合同**：填写真实研究区间、历史币池依据、数据既往使用情况及统计规则。A约2年、B约1年、C约1年的方向保留，正式日历日期尚未冻结；A段不填写轮数或候选数量额度。
- **逐小时币池**：CSV包含 `timestamp,symbol,eligible` 三列。时间使用UTC小时K线的开盘标签；每小时明确每个资产的true/false，涵盖预热期。A/B使用相同资产范围的完整网格，新上市币在之前的时点明确为false。
- **模型和登录**：完成 `codex login` 的 ChatGPT 登录；默认使用 `gpt-5.6-luna`。`--model` 和 `--reasoning-effort` 可显式选择，启动时检查账号模型目录。

[示例合同](../../examples/factor_mining/contract.example.json) 和 [示例币池](../../examples/factor_mining/universe.example.csv) 是短区间、固定六币的**工程样例**，会被程序阻止进入创意库。里面的参数不代表正式研究参数已经确认。

2026-09-28起，新合同可沿用`admission_scheme="plan3"`并显式设置`plan3_tracks_gate=false`、`fdr_method="BH"`：B段只把三轨作为标签，准入仍要求整批Rank IC校正通过、达到合同的`min_abs_ic`及有向平均价差大于0。公式不再设置`expression_policy`，编译器不推导单位或要求无量纲，也不拦截原始价格和基础币数量的跨币排名；仍检查白名单、参数、AST复杂度、回看和数据可用时间。示例合同展示Plan3标签开关及工程用的0.010 IC下限；正式研究须使用新的运行ID和明确的数据使用说明。既有冻结运行及证据不改写。

## Goal模式：持续推进成果目标

优化Agent负责整体研究任务和候选优化，程序负责持续调度；未完成目标时，空优化提案不会结束整个Goal。目标文本在启动时填写，B验收沿用研究合同。第一版的可核验成果是合格因子创意卡，必须明确指定目标数量；没有默认“找到1个”。

准备自己的Goal JSON，例如下方只是文件格式示例，目标和数量均需按实际研究填写：

```json
{
  "goal_id": "my-factor-goal",
  "objective": "研究指定方向，获得符合目标要求、通过既有B验收的因子创意卡",
  "target_ideas": 2
}
```

启动时传入已经准备好的正式研究合同和币池。下面的`path/to/...`是待替换的文件路径：

```bash
.venv/bin/python -m crypto_quant.cli factor-mine goal-start \
  --goal path/to/goal.json \
  --contract path/to/research-contract.json \
  --universe path/to/universe.csv \
  --model gpt-5.6-luna --reasoning-effort max

# 查询进度：不调用模型、不读取行情
.venv/bin/python -m crypto_quant.cli factor-mine goal-status \
  --goal-dir experiments/factor_mining/goals/my-factor-goal

# Ctrl+C暂停后恢复，复用已保存的目标、输入路径和模型设置
.venv/bin/python -m crypto_quant.cli factor-mine goal-resume \
  --goal-dir experiments/factor_mining/goals/my-factor-goal
```

- **继续优化**：沿用候选四问、修改任务和配对检验。
- **探索新方向**：当前候选结束后，优化Agent依据A历史安排新任务，构想Agent生成具体公式。
- **等待数据**：优化Agent声明字段、最低有效A观测数和依据；程序默认每60秒检查，满足后自动继续。`--poll-seconds`只调整等待检查间隔，不限制研究时长。需要在保存的数据库与币池路径补齐数据；Goal不自行采集或改变研究区间。

只有程序准入、创意卡完整、优化Agent确认与Goal文本匹配的成果才计数。目标数量达到后进入`complete`。工程合同不会产生可计入目标的成果；不要把工程示例直接作为正式成果Goal长期运行。

状态与每次转移保存在Goal目录`events/`；研究上下文、成果核验和各次运行分别留档。`waiting`会继续检查依赖，`paused`表示用户中断，`error`表示具体故障待处理；都不等于完成。恢复时先核验代码和数据，复用已提交的A检查点；B已完成数值结果时只补解释，数值未完成则报错要求检查原证据，不再次读取B。已完成Goal再次恢复不会新发模型请求。

B详细数值和失败原因不传给构想或优化研究上下文；成果核验上下文仅获得程序准入凭据与对应A证据，不用于安排下一方向。既有B标准、用途限制和C隔离保持不变。

## 操作顺序

以下命令会调用所选真实模型；先完成上面的配置。示例不设置单次输出token上限（`output_tokens=null`），本地输入上下文预算为80万字节上界，不设总请求次数、A段轮数或候选数量上限；默认使用max推理强度。此前16384输出上限导致截断的记录保留在历史验证中。

```bash
# A段探索：使用官方Codex SDK和ChatGPT订阅
.venv/bin/python -m crypto_quant.cli factor-mine explore \
  --contract examples/factor_mining/contract.example.json \
  --universe examples/factor_mining/universe.example.csv \
  --model gpt-5.6-luna --reasoning-effort max

# 从a-complete.json的retained_ids选择实际候选ID；为空则没有可冻结候选
.venv/bin/python -m crypto_quant.cli factor-mine freeze \
  --run-dir experiments/factor_mining/no_units_tags_example_20260928 \
  --candidate-ids candidate-0001 \
  --universe examples/factor_mining/universe.example.csv

# 冻结批次的B段验证
.venv/bin/python -m crypto_quant.cli factor-mine validate \
  --run-dir experiments/factor_mining/no_units_tags_example_20260928 \
  --universe examples/factor_mining/universe.example.csv \
  --model gpt-5.6-luna --reasoning-effort max
```

每个运行ID对应独立目录，已有研究证据不会被覆盖。B段开始读取前即记录使用状态，同一运行不能再次读取B验证或回到A改公式。回复错误按上方规则纠正、隔离或停止；已保存完整数值的解释可用`complete-reports`单独补齐。该入口不受累计调用次数限制，也不恢复构想或优化循环。

研究运行不再绑定代码指纹。旧运行报告仍可查阅；继续冻结或验证时，仍须通过合同、输入和证据检查。上方2026-09-14联调属于旧版本，不代表新决策流程已完成真实模型验收。

根目录新环境已包含绘图、清算数据和测试依赖。完整验证命令（同时收集unittest及pytest测试）：

```bash
.venv/bin/python -m pytest -q
```

实现规则及证据边界见 [模块说明](../../src/crypto_quant/research/factor_mining/README.md)。
