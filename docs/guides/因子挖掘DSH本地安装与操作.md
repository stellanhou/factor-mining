# DSH 因子研究：本地安装与操作

适用版本：DeepSeek Harness `0.2.1-rc.2`，研究 bundle `0.3.0`。本轮只使用本地合成资料；验收状态与失败历史见[工程交付报告](../reports/DSH本地工程交付与验收_20261018.md)。

## 安装

在项目目录准备不含 Codex SDK 的环境，保留已有环境与配置：

```sh
UV_PROJECT_ENVIRONMENT="$PWD/.venv-dsh" uv sync --locked --extra native-chats --extra factor-mining
mkdir -p dist/dsh
npm pack ./harness/factor-research --ignore-scripts --pack-destination ./dist/dsh
export DSH_HOME="$PWD/experiments/factor_mining/dsh-home"
```

第一次初始化专用 profile。下面的存在性检查失败时，应使用已有 profile，不执行重建：

```sh
test ! -e "$DSH_HOME/profiles/factor-research"
dsh --profile factor-research --from-default-profile web --dump-config
dsh plugin --profile factor-research add "$PWD/dist/dsh/factor-research-dsh-bundle-0.3.0.tgz" --ignore-scripts
```

`--from-default-profile` 会重建目标 profile，不能对已有研究 profile 重复执行。包由官方 CLI 安装；源码链接只用于开发。DSH 提供共享的运行时依赖，无需把第二份 DSH/Cordis 打进 bundle。

通过 DSH 自己的账户/模型设置配置本研究 profile。验收使用 `deepseek-account / deepseek-flash / high`；不要修改个人全局模型、复制登录文件或在研究对话中粘贴凭据。

## 绑定和启动

启动需要一个已准备、已核验的运行绑定和 `run.patch.yml`。绑定包含固定目标、合同、合成数据与币池引用、三个不同的 Session ID；运行补丁包含同样的身份、Workspace、有限请求预算、输出上限和停止超时。不能用历史 Codex thread ID 代替 DSH Session ID。

首次使用新 Harness home 时，通过 DSH 创建/选择专用 Workspace，再把真实 Workspace ID 写入本次运行补丁。既有验收夹具和其 Session 不得当作新研究重置使用。

```sh
export FACTOR_RESEARCH_PROJECT_ROOT="$PWD"
export FACTOR_RESEARCH_PYTHON="$PWD/.venv-dsh/bin/python"
export FACTOR_RESEARCH_WORKSPACE_ID="<本次 Workspace ID>"
export FACTOR_RESEARCH_BINDING="<本次 binding.json 绝对路径>"
export FACTOR_RESEARCH_RUN_ROOT="<本次研究目录绝对路径>"
dsh --profile factor-research --patch /absolute/path/run.patch.yml --dump-config > /absolute/path/final-config.yml
dsh --profile factor-research --patch /absolute/path/run.patch.yml --host 127.0.0.1 --port 49922
```

检查最终组合配置，包括 bundle、profile、home 和启动覆盖层。必需控制插件、三角色身份、MCP 服务、预算和只读界面必须一致；HMR、通用执行/文件工具、自动标题模型请求和 provider 自动重试保持关闭。原生压缩启用，摘要请求也计入同一总预算。研究 MCP 单次工具调用上限为300秒；停止仍通过独立控制通道取消，不等待工具超时。

不要运行旧的固定 Codex 聊天夹具来初始化新的 DSH 研究。`scripts/prepare_native_chats_synthetic_acceptance.py` 是历史领域夹具，不是通用 DSH 建项命令。

## 日常研究

- 在“研究协调”输入目标和问题。“A 研究”“B 验证”按钮用于只读观察；服务端也拒绝对 A/B 直发、改队列、改角色、fork 和直接人类命令。
- 已绑定目标完整时，在协调处要求启动。A 自主提出候选、补读证据、预声明修改并检验；协调通过原生回合事件收取结果，不需要人工搬运报告或轮询。
- 运行中直接向协调询问“当前阶段、在途动作、已有证据和下一步”。普通提问不会改变合同。
- A 完整封存后，协调冻结全部保留候选并派发唯一 B 工单。B 只计算一次；后续只读取检查点、解释和交付。
- 程序 `objective_met=true` 且完整交付时，协调才能完成同一个原生 Goal。零保留、B 拒绝或交付不足会保留缺口并暂停；不能把“流程已结束”当作达标。

“暂停目标”只暂停 Goal。需要取消模型与计算时，点击“停止研究”，并等界面确认三会话和计算已收敛。

重启沿用原绑定、原 Session、工单与动作 ID。恢复后持久 Goal 即使是 active，也可能处于 disarmed；须点击“恢复目标”。显式恢复通知允许对账先前用户中断的动作，不会重发已经接受的研究任务。已完成 Goal 不会自动重启。

每回合请求上限触发后，协调只可使用最多三次、按顺序的已验证续传，或基于变化后的 A 快照准备恢复工单。原始 A 动作参数通过 `read_a_action_request` 从本 A Session 的原始调用读取，并与领域回执 hash 核对；不能猜测参数或把同一个 action_id 用于另一工具。

协调自身达到每回合上限时，Host只针对该确切回合发出一次续接通知，沿用已存结果；次数受绑定Goal轮数上限及总预算约束。供应商/持久化错误不走这条路径。

总请求预算耗尽会拒绝新的 provider 请求并暂停活跃 Goal。不要通过重启、更换运行目录或另建 Goal 重置账本。B 已访问但没有完整检查点时停止，不重载 B。

## 在封存前批准合同变更

变更必须来自人类明确批准。协调的普通文字建议不会自行改合同。原生人类命令位于同一个协调输入框：

```text
/research-change {"format_version":1,"child_research_id":"approved-change-v2","original_intent":"提高最低IC门槛。","adopted_change":"最低绝对IC提高到0.03，其余边界不变。","new_objective":"填写完整的新研究目标。","contract_changes":{"min_abs_ic":0.03},"goal_changes":{}}
```

该命令不进入模型消息历史，也不是模型工具。它先停止旧运行，检查 A 未封存、无冻结/B 访问、无未对账动作，再保存原意、原/新合同版本、旧任务和 A 证据引用。旧文件不覆盖；三个 Session 身份保留，原生 Goal 只编辑版本，不新建 Goal。

命令保存新绑定和子运行补丁，并把旧版本标为已被替代。按保存的子运行信息重启：保留原 `run.patch.yml`，在其后追加 `child.patch.yml`，将 `FACTOR_RESEARCH_BINDING` 和 `FACTOR_RESEARCH_RUN_ROOT` 指向新版本。子补丁的 `modelBudget.budgetRoot` 继续指向原账本，因此预算不重置。点击恢复后，让协调读取新版本目标并执行。

已封存、已冻结或 B 已访问的研究不能通过变更重开 A。若存在未对账动作，先用原动作和原参数恢复，完成后再批准变更。

## 用量与验收边界

DSH 日志与 `.factor-research-control/model-budget/reservations/` 共同记录实际请求。摘要请求有 `purpose=compaction`；请求预留可能对应失败或中断，不等于成功回答数。原生回执、领域回执、A/B 检查点分别核对。

累计输入 token/费用硬上限与费用计量尚不可用，不填写为零。本轮合成准入通过只证明工程链路，不能作为真实市场因子有效性结论。
