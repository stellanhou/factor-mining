# 因子挖掘全量运行复盘：Goal-v22 运行中断与 Optimizer 决策死锁

- **记录日期**：2026-09-19
- **关联运行**：`experiments/factor_mining/goals/full-history-five-ideas-20260917-v22`
- **运行配置**：30 资产全量历史面板，A 段 2026-04-01 至 2026-07-12（2447 小时），B 段 2026-07-13 至 2026-07-31（独立冻结验收），Goal 目标 5 个通过 B 验收的因子创意卡。
- **关联代码**：`src/crypto_quant/research/factor_mining/workflow.py`, `goal.py`

---

## 一、 运行总体概况

在 2026-09-18 凌晨的 Goal-v22 全量连续自主运行中，多 Agent 因子挖掘系统完成了 **24 个完整研究循环（Cycle 1 ~ 24）**，累计构想、清洗、执行与评估了 **103 个因子候选**，并在 A 段实际挖掘出了多个高 Rank IC 且统计显著的因子。

然而，本次运行暴露出两大致命问题：
1. **技术阻塞**：在 Cycle 25 遭遇底层 AST 回看解析异常，耗尽纠错次数后抛出 `ModelResponseError` 中断运行；
2. **机制死锁**：在 Cycle 1 ~ 24 中，Optimizer Agent 因 Prompt 过度防御，产生 **102 次 pause、3 次 optimize、1 次 discard、0 次 retain**，导致没有任何候选被送入 B 段独立验证，系统空转 24 轮未产出任何创意卡。

---

## 二、 问题一：预计算特征回看 AST 校验崩溃

### 1. 故障现象与错误堆栈
在 Cycle 25（Run: `goal-84ac74aa6c0e-000025`），Ideator 构想第 4 个候选 `spot_perp_basis_funding_state`：
- **表达式**：`mul(cross_rank(sub(div(perp_close, spot_close), 1)), neg(cross_zscore(funding_7d_sum)))`
- **自然语言含义**：`"...再取过去168小时实际结算费率之和 funding_7d_sum...最大回看168小时。"`
- **系统报错**：
  ```text
  crypto_quant.research.factor_mining.records.ModelResponseError: 
  ValueError: stated maximum lookback differs from executable expression: claimed=[168], actual=0
  ```
  模型在随后的两次回复纠正（Correction 1 & 2）中未能理解报错机制，最终触发孤立故障保护并导致任务终止。

### 2. 根本原因剖析
- `_reject_misstated_lookback` 会调用 `compile_expression(expression).lookback_hours` 提取表达式树中的时序回看。
- `compile_expression` 仅递归累加显式 `ts_*` 时序算子（如 `ts_delay`、`ts_corr` 等）的参数。
- 特征 `funding_7d_sum` 与 `funding_24h_sum` 是直接从数据底座加载的**预计算特征列**（Precomputed Input Fields），在表达式 AST 中作为叶子节点变量名（`ast.Name`），其 AST 算子回看值为 0。
- 模型在理解业务金融逻辑时，认为该特征涵盖了过去 7 天（168 小时）的资金费，因此在自然语言中如实声明“最大回看168小时”。
- 校验函数将业务层面的回看与 AST 算子回看直接做严格相等性比较（`claimed == actual`），导致误杀。

### 3. 修复方案与验证
- **修复逻辑**：在 `_reject_misstated_lookback` 中引入预计算特征的内生时间窗口映射（`funding_7d_sum: 168`, `funding_24h_sum: 24`）。当表达式包含此类字段时，允许声明回看等于 `actual`（纯算子回看）或 `actual + inherent`（含内生窗口的总回看）。
- **Prompt 引导**：在 Ideator 提示词中明确补充说明预计算字段与算子回看的关系，消除口径歧义。
- **验证结论**：新增单元测试验证覆盖 0h 与 168h 均合法通过、非法值（如 50h）严格拒绝；全量 84 项 factor mining 测试与 19 项 goal 测试全部通过。

---

## 三、 问题二：Optimizer 决策死锁（全员 Pause，0 候选送 B）

### 1. 现象与真实数据
在 A 段评估的 103 个因子中，出现了一批统计表现非常优异的因子候选，例如：
- `spot_perp_range_per_volume_impact_s`（Cycle 17）：Rank IC = **-0.0421**（$p = 0.0399$）
- `mark_index_basis_change_24h_ex_leve`（Cycle 14）：Rank IC = **-0.0365**（$p = 0.0018$），$95\%$ 置信区间不跨零，正收益阶段占比 $73\%$
- `toptrader_position_account_net_dive`（Cycle 22）：Rank IC = **-0.0346**（$p = 0.0489$）
- `mark_index_basis_change_24h_rank`（Cycle 14）：Rank IC = **-0.0221**（$p = 0.0055$）

但在 103 次候选决策中，Optimizer 给出的去向分布为：
- `pause`：102 次
- `optimize`：3 次
- `discard`：1 次
- **`retain`：0 次**

### 2. 真实决策日志剖析
以 Cycle 14 中的高表现候选 `candidate-0003` 为例，Optimizer 实际给出的决策理由如下：
> *"candidate-0003 是本轮方向与预注册一致且 rank_ic 点估计略高于门槛的候选，程序语义一致；但 directional_spread 的 95% CI 跨零，分组非单调……**未做 BY FDR、正交化和分页核验，A 段含先前开发检查日期，不是独立验证样本**。其修改路线已暂停且未确认配对增量。**当前版本不足以固定送 B，也无具体新修改路线，故 pause 保留**。"*

### 3. 根因剖析：A/B 段防过拟合机制的职责错位
查看原有 Optimizer Prompt：
```text
"保留须解释A证据为何值得固定版本送B，A阶段不宣称通过独立验证；"
"证据不足则pause并写恢复条件，有充分否定证据或无增量的重复才discard。"
"不得仅因低分或不显著而淘汰。"
```

大模型被引导进入了一个**过度防御的死循环**：
1. **概念错位**：大模型把“未做全批次 BY FDR 检验”、“A 段不是独立样本”作为了**拒绝送 B 的理由**。
2. **逻辑悖论**：在量化架构中，**全批次 BY FDR 校准和独立样本检验恰恰是 B 段的职责**！A 段是自适应探索期，任何因子在 A 段必然不是“独立样本”。
3. **后果**：如果要求因子必须在 A 段就证明自己“通过了独立验证与全批次 FDR”才能给 `retain`，那么逻辑上没有任何候选能够拿到进入 B 段的准考证；系统每一轮结算时 `retained_ids` 永远为空，自动略过 B 段验收，直接空转进入下一轮。

---

## 四、 核心反思与修复实施

1. **澄清 `retain` 的语义**：
   - `retain` **绝不等于**“该因子已被彻底证实有效/可以直接实盘”。
   - `retain` **代表且仅代表**：“该因子在 A 段探索期表现出统计显著性、单调性与自洽机制，**具备进入 B 段样本外冻结验收的‘准考证资格’**”。
2. **建立明确的送 B（retain）引导准则**：
   - 告知 Optimizer：A 段不需要也不可能完成独立样本验证或全批次 FDR（那是 B 段的程序化验收职责）；
   - 当因子在 A 段满足基本统计门槛（如名义 $|Rank\ IC| \ge 0.02$ 且 $p < 0.05$、方向一致、机制合理）时，**应当积极赋予 `retain`** 送入 B 段冻结批次。
   - B 段通过了，生成创意卡；B 段未通过，程序直接记录未通过原因并存档，绝不返工污染。
3. **已实施修改（2026-09-19）**：
   - **AST 回看校验修复**：在 `workflow.py` 的 `_reject_misstated_lookback` 中引入预计算特征的时间窗口映射，并在 Ideator prompt 中消除口径歧义。
   - **Optimizer Prompt 升级**：在 `workflow.py` 中明确区分 A 段探索与 B 段验收职责，禁止将 A 段非独立样本或未做 FDR 作为拒绝送 B 的理由；设立正向 `retain` 准则（$|IC| \ge 0.02$ 且 $p < 0.05$、方向一致、机制自洽）。
   - **回归测试验证**：全量 139 项单元测试（涵盖 84 项 factor mining 测试与 19 项 goal 测试）100% 通过（42.2 秒）。
