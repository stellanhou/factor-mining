"""Readable evidence reports with static grouped-return plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_group_plot(path: Path, report: dict[str, Any]) -> None:
    from matplotlib.figure import Figure
    from matplotlib.ticker import PercentFormatter

    means = report["summary"]["group_means"]
    figure = Figure(figsize=(7, 3.5), layout="constrained")
    axis = figure.subplots()
    x = list(range(1, len(means) + 1))
    for index, value in enumerate(means.values(), 1):
        if value is not None:
            axis.bar(index, value, color="#247b84" if value >= 0 else "#c06446", width=0.65)
        else:
            axis.annotate("N/A", (index, 0), ha="center", va="bottom")
    axis.axhline(0, color="#666666", linewidth=0.7)
    axis.set_xticks(x, [f"Group {i}" for i in x])
    axis.set_xlabel("Factor value: low to high (equal values remain together)")
    axis.set_ylabel("Mean 24-hour forward return")
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.set_title(f"Segment {report['segment']} | fixed direction {report['direction']:+d}")
    axis.spines[["top", "right"]].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        figure.savefig(handle, format="svg", metadata={"Date": None})


def write_research_report(path: Path, run_id: str, purpose: str, records: list[dict[str, Any]], stage: str,
                          *, replace: bool = False) -> None:
    lines = [f"# 因子挖掘 {run_id}：{stage} 段", "",
             "用途：工程验证，不能作为真实预测能力证据。" if purpose == "engineering_check" else "用途：候选因子研究，后续仍需完整策略验证。", "",
             "原始记录、精确模型请求和回复均保存在本运行目录内。统计事实与模型解释分别展示。", ""]
    reports = {r["data"]["candidate_id"]: r["data"] for r in records if r["kind"] == "model_report"}
    format_notes = [r for r in records if r["kind"] == "format_deviation"]
    validations = {r["data"]["candidate_id"]: r for r in records if r["kind"] == "validation_result"}
    pending = [r["data"]["candidate_id"] for r in records
               if r["kind"] == "evaluation" and r["data"]["candidate_id"] not in reports]
    if pending:
        lines += ["## 待补全解释", "", "数值结果已保存，以下候选尚缺合格的模型解释：" + "、".join(pending) + "。",
                  "可使用 complete-reports 补全；该操作只读取已有研究记录。", ""]
    failures = [r for r in records if r["kind"] in {"invalid_model_response", "report_pending"}]
    if failures:
        lines += ["## 回复纠正与处理历史", ""]
        for record in failures:
            data = record["data"]
            label = data["role"] if record["kind"] == "invalid_model_response" else data["candidate_id"]
            reason = data["error"] if record["kind"] == "invalid_model_response" else data["reason"]
            lines.append(f"- {label}：{reason}。[{record['id']}]({stage.lower()}_records/{record['id']}.json)")
        lines += [""]
    if format_notes:
        lines += ["## 模型回复格式记录", "", "以下附加字段已留痕，仅声明的核心字段用于后续流程。", ""]
        for note in format_notes:
            data = note["data"]
            fields = "、".join(f"`{field}`" for field in data["extra_fields"])
            lines.append(f"- {data['role']}：{fields}。"
                         f"[格式记录]({stage.lower()}_records/{note['id']}.json) · [原始回复]({data['raw_response']})")
        lines += [""]
    if stage == "A":
        decisions = {}
        proposals = []
        for record in records:
            if record["kind"] == "optimization":
                decisions.update({d["candidate_id"]: d for d in record["data"]["decisions"]})
                proposals.extend((record["id"], proposal) for proposal in record["data"]["proposals"])
        labels = {"optimize": "继续优化", "retain": "保留待B验证", "pause": "暂停", "discard": "淘汰"}
        questions = {"research_basis": "有研究依据", "modification_hypothesis": "有具体修改假设",
                     "verifiable_improvement": "能验证改善", "attempt_value": "还有尝试价值"}
        if decisions:
            lines += ["## 候选去向", "", "以下为优化 Agent 的研究决定，B验证及创意卡准入另行执行。", ""]
        for cid, decision in decisions.items():
            lines += [f"### {cid}：{labels[decision['disposition']]}", "",
                      f"继续修改：{'是' if decision['continue_optimization'] else '否'}。", "", decision["reason"], ""]
            for key, answer in decision["answers"].items():
                lines.append(f"- {questions[key]}：{'支持' if answer['supported'] else '不支持'}。{answer['reason']}")
            lines += ["", "依据：" + "、".join(f"[{ref}](a_records/{ref}.json)" for ref in decision["evidence_refs"]), ""]
            if decision["resume_condition"] is not None:
                lines += ["恢复条件：" + decision["resume_condition"], ""]
        if proposals:
            lines += ["## 修改任务与配对检验合同", "",
                      "Optimizer只声明修改目标和判断规则；最终候选定义及公式由下一轮Ideator生成。", ""]
            for record_id, proposal in proposals:
                task, design = proposal["modification_task"], proposal["experiment_design"]
                refs = "、".join(f"[{ref}](a_records/{ref}.json)" for ref in proposal["evidence_refs"])
                lines += [f"### {proposal['proposal_id']} → {proposal['control_id']}", "",
                          f"- 原核心假设：{task['core_hypothesis']}",
                          f"- 已观察问题：{task['observed_problem']}",
                          f"- 修改假设：{task['modification_hypothesis']}",
                          f"- 改动目标：{task['change_target']}",
                          f"- 固定部分：{task['fixed_components']}",
                          f"- 配对问题：{design['question']}",
                          f"- 判断标准：{design['metric']}最低改善{design['min_improvement']}，"
                          f"允许有向IC损失{design['max_ic_loss']}",
                          f"- 预期结果：{design['expected_outcome']}",
                          f"- 停止条件：{design['stop_condition']}",
                          f"- 暂停条件：{design['pause_condition']}",
                          f"- 依据：{refs}", "",
                          f"[完整建议](a_records/{record_id}.json)", ""]
    for record in records:
        if record["kind"] != "evaluation":
            continue
        data = record["data"]
        cid = data["candidate_id"]
        summary = data["summary"]
        lines += [f"## {cid}", "", "| 指标 | 数值 |", "|---|---|",
                  f"| 平均 Rank IC | {summary['rank_ic']['mean']} |",
                  f"| ICIR（未年化） | {summary['rank_ic']['mean_std_ratio']} |",
                  f"| 有向高低组收益差 | {summary['directional_spread']['mean']} |",
                  f"| 有效 IC 期数 | {summary['rank_ic']['n']} |",
                  f"| 跨段排除小时数 | {data['coverage']['purged_hours']} |", "",
                  f"![各组未来24小时平均收益](plots/{cid}-{stage}.svg)", ""]
        if cid in reports:
            model = reports[cid]
            lines += ["### 模型解释", "", model["analysis"], "", "适用及失效条件：", ""]
            lines += [f"- {value}" for value in model["conditions"]]
            lines += ["", "证据限制：", ""] + [f"- {value}" for value in model["limitations"]] + [""]
        if cid in validations:
            validation = validations[cid]
            outcome = validation["data"]
            lines += ["### B段程序验证结果", "",
                      f"预定验证规则：{'通过' if outcome['validation_status'] == 'passed' else '未通过'}。",
                      f"创意卡交付资格：{'满足' if outcome['eligible_for_idea_pool'] else '不满足'}。", ""]
            lines += [f"- {reason}" for reason in outcome["reasons"]]
            lines += ["", f"[逐项程序判定](b_records/{validation['id']}.json)。模型解释不改变此结果。", ""]
        lines += [f"完整逐期、分阶段、分组和不确定性结果见 `{stage.lower()}_records/{record['id']}.json`。", ""]
    unsuccessful = [r for r in records if r["kind"] == "duplicate"
                    or (r["kind"] == "calculation" and r["data"]["status"] != "computed")]
    if unsuccessful:
        lines += ["## 失败、重复与待补充定义", ""]
        for record in unsuccessful:
            data = record["data"]
            lines.append(f"- {data['candidate_id']}：{data.get('status', record['kind'])}，见 `{stage.lower()}_records/{record['id']}.json`。")
    with path.open("w" if replace else "x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
