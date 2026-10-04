"""Readable evidence reports with static grouped-return plots."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any


def _cell(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, list):
        return "[" + ", ".join(_cell(item) for item in value) + "]"
    if isinstance(value, (int, float)):
        return f"{value:.6g}"
    return str(value)


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
    horizon_hours = report["horizon_hours"]
    axis.set_ylabel(f"Mean {horizon_hours}-hour forward return")
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.set_title(f"Segment {report['segment']} | {horizon_hours}h | fixed direction {report['direction']:+d}")
    axis.spines[["top", "right"]].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        figure.savefig(handle, format="svg", metadata={"Date": None})


def _archive_file(run_dir: Path, pointer: dict[str, Any]) -> Path:
    root = pointer["root"]
    identity = pointer["identity"]
    if not isinstance(root, str) or Path(root).is_absolute() or not isinstance(identity, dict):
        raise ValueError("因子归档定位信息无效")
    archive_root = (run_dir.resolve() / root).resolve()
    registry = archive_root / "registry.sqlite3"
    if not registry.is_file():
        raise FileNotFoundError(registry)
    try:
        connection = sqlite3.connect(f"{registry.as_uri()}?mode=ro", uri=True)
        try:
            row = connection.execute("""SELECT factor_id FROM factor_identities
                WHERE expanded_expression=? AND direction=? AND semantics_version=?""",
                (identity["expanded_expression"], identity["direction"],
                 identity["semantics_version"])).fetchone()
        finally:
            connection.close()
    except (KeyError, sqlite3.DatabaseError) as exc:
        raise ValueError(f"因子归档注册表无效：{registry}") from exc
    if row is None:
        raise ValueError(f"因子归档身份不存在：{registry}")
    path = archive_root / f"factor-{int(row[0]):06d}.sqlite3"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _rank_displacement_section(lines: list[str], run_dir: Path, report: dict[str, Any] | None,
                               locator: dict[str, Any] | None) -> None:
    lines += ["### 信号持续性与换手倾向代理", ""]
    if report is None:
        if locator is not None:
            raise ValueError("rank-displacement archive locator has no diagnostic report")
        lines += ["未计算：该次运行没有保存排名变化诊断。", ""]
        return
    if locator is None:
        raise ValueError("rank-displacement report has no archive locator")
    deltas = report["deltas"]
    lines += [f"定义版本：`{report['definition_version']}`；评价段：{report['segment']}；"
              f"统计网格：每 {report['sample_hours']} 小时。", "",
              "排名变化 D 描述共同合格币种的百分位排序变化；该指标本身不代表手续费或净收益。", "",
              "| 间隔 Δ | 平均 D | 中位数 | P90 | 有效时点 / 应评时点 | 有效比例 | 边界排除 | 缺精确小时 | 共同币种不足 |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for delta in ("1", "4", "24"):
        result = deltas[delta]
        summary, coverage = result["summary"], result["coverage"]
        lines.append("| " + " | ".join(_cell(value) for value in (
            f"{delta}h", summary["mean"], summary["median"], summary["p90"],
            f"{summary['valid_periods']} / {summary['expected_periods']}",
            summary["valid_period_share"], coverage["boundary_excluded_periods"],
            coverage["missing_exact_hour_periods"],
            coverage["insufficient_common_symbol_periods"])) + " |")
    lines += ["", "各阶段平均 D："]
    for delta in ("1", "4", "24"):
        values = [f"{stage['start']} 至 {stage['end']}：{_cell(stage['mean'])}"
                  for stage in deltas[delta]["stages"]]
        lines.append(f"- Δ={delta}h：" + ("；".join(values) if values else "无有效阶段"))
    archive_path = _archive_file(run_dir, locator)
    relative_archive = Path(os.path.relpath(archive_path, run_dir.resolve())).as_posix()
    lines += ["", f"逐时指标、阶段明细和覆盖记录：SQLite evaluation `{locator['evaluation_id']}`，"
              f"[排名变化诊断归档]({relative_archive})。", ""]


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
            if decision["disposition"] == "retain":
                lines.append("- 保留送检期限：" + "、".join(
                    f"{h}h" for h in decision["retained_horizons"]))
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
        displacement_comparisons = [record for record in records
                                    if record["kind"] == "experiment_result"
                                    and record["data"].get("plan", {}).get("experiment_design", {}).get("metric")
                                    == "rank_displacement"]
        if displacement_comparisons:
            lines += ["## 排名变化受控修改配对结果", "",
                      "程序在原候选与修改版的共同币种、时点和有效标签上重算 D 与方向 Rank IC。", "",
                      "| 候选 / 对照 | H / Δ | D 改善均值及区间 | 有向 IC 变化及区间 | 配对有效时点 | 路线程序状态 |",
                      "|---|---:|---:|---:|---:|---|"]
            for record in displacement_comparisons:
                data = record["data"]
                coverage = data["coverage"]
                improvement, ic_change = data["paired_improvement"], data["paired_ic_change"]
                h_delta = f"{data['horizon_hours']}h / {data['displacement_hours']}h"
                paired_periods = (f"{coverage['paired_valid_periods']} / {coverage['expected_periods']}"
                                  f"（{_cell(coverage['paired_valid_period_share'])}）")
                lines.append("| " + " | ".join(_cell(value) for value in (
                    f"{data['candidate_id']} / {data['plan']['control_id']}", h_delta,
                    f"{_cell(improvement['mean'])} [{_cell(improvement['ci'])}]",
                    f"{_cell(ic_change['mean'])} [{_cell(ic_change['ci'])}]",
                    paired_periods, data["route_decision"])) + " |")
                lines += ["", f"{data['candidate_id']}：{data['reason']}。",
                          f"主指标决定：{data['decision']}；路线状态：{data['route_decision']}。",
                          f"覆盖：边界排除 {coverage['boundary_excluded_periods']} 期，"
                          f"缺精确小时 {coverage['missing_exact_hour_periods']} 期，"
                          f"共同币种不足 {coverage['insufficient_common_symbol_periods']} 期。",
                          f"[逐期配对证据与合同](a_records/{record['id']}.json)", ""]
    for record in records:
        if record["kind"] != "evaluation":
            continue
        data = record["data"]
        cid = data["candidate_id"]
        lines += [f"## {cid}", ""]
        if stage == "A":
            primary = data
            primary_summary = primary["summary"]
            primary_horizon = primary["horizon_hours"]
            lines += [f"A段主评估期限：{primary_horizon}h；以下主评估数值使用全主评估样本。", "",
                      "| 指标 | 数值 |", "|---|---|",
                      f"| 平均 Rank IC | {primary_summary['rank_ic']['mean']} |",
                      f"| ICIR（未年化） | {primary_summary['rank_ic']['mean_std_ratio']} |",
                      f"| 有向高低组收益差 | {primary_summary['directional_spread']['mean']} |",
                      f"| 有效 IC 期数 | {primary_summary['rank_ic']['n']} |",
                      f"| 跨段排除小时数 | {primary['coverage']['purged_hours']} |", ""]
            if "horizon_comparison" in primary:
                horizons = primary["horizon_comparison"]["horizons"]
                horizon_ids = sorted(int(h) for h in horizons)
                summaries = [horizons[str(h)]["summary"] for h in horizon_ids]
                lines += ["### A段共同样本期限对照", "",
                          "各期限使用相同因子值、方向、共同有效时点和币种，并排除跨段标签。"
                          "各组等权；收益未扣交易成本和资金费。", "",
                          "| 指标 | " + " | ".join(f"未来{h}h" for h in horizon_ids) + " |",
                          "|---|" + "---|" * len(horizon_ids)]
                rows = [
                    ("平均 Rank IC", [s["rank_ic"]["mean"] for s in summaries]),
                    ("Rank IC 标准误", [s["rank_ic"]["se"] for s in summaries]),
                    ("Rank IC 置信区间", [s["rank_ic"]["ci"] for s in summaries]),
                    ("Rank IC 原始 p 值", [s["rank_ic"]["p_value"] for s in summaries]),
                    ("固定方向平均 Rank IC", [None if s["rank_ic"]["mean"] is None else s["rank_ic"]["mean"] * horizons[str(h)]["direction"]
                                             for h, s in zip(horizon_ids, summaries)]),
                    ("ICIR（未年化）", [s["rank_ic"]["mean_std_ratio"] for s in summaries]),
                    ("有向高低组收益差", [s["directional_spread"]["mean"] for s in summaries]),
                    ("价差置信区间", [s["directional_spread"]["ci"] for s in summaries]),
                    ("IC方向一致期占比", [s["ic_direction_share"] for s in summaries]),
                    ("正向阶段占比", [s["positive_stage_share"] for s in summaries]),
                    ("有效阶段数", [s["valid_stages"] for s in summaries]),
                    ("有效 IC 期数", [s["rank_ic"]["n"] for s in summaries]),
                    ("有效资产小时观测", [horizons[str(h)]["coverage"]["eligible_observations"]
                                        for h in horizon_ids]),
                    ("跨段排除资产小时观测", [horizons[str(h)]["coverage"]["purged_observations"]
                                          for h in horizon_ids]),
                    ("跨段排除小时数", [horizons[str(h)]["coverage"]["purged_hours"]
                                        for h in horizon_ids]),
                ]
                rows += [(f"第{group}组平均收益", [s["group_means"][group] for s in summaries])
                         for group in summaries[0]["group_means"]]
                for label, values in rows:
                    lines.append("| " + label + " | " + " | ".join(_cell(value) for value in values) + " |")
                lines += ["", "对照表沿用相同 HAC 带宽，p 值为未校正的开发诊断；它保留每个期限的不确定性、滚动及分阶段结果，不自动挑选最佳期限。", "",
                          "各期限分组收益图：" + "、".join(
                              f"[{h}h](plots/{cid}-A-{h}h.svg)" for h in horizon_ids), ""]
        else:
            horizons = data["horizons"]
            horizon_ids = sorted(int(h) for h in horizons)
            outcome = validations[cid]["data"]
            horizon_results = outcome["horizon_results"]
            tests = {test["horizon_hours"]: test for test in outcome["tests"]}
            correction = next(r["data"] for r in records if r["kind"] == "multiple_testing")
            summaries = [horizons[str(h)]["summary"] for h in horizon_ids]
            lines += ["### B段逐期限程序结果", "",
                      f"各期限使用冻结的 A 保留范围和对应段边界；BH 家族包含 {correction['family_size']} 个候选 × 期限检验，"
                      f"FDR α={correction['alpha']}。", "",
                      "| 指标 | " + " | ".join(f"未来{h}h" for h in horizon_ids) + " |",
                      "|---|" + "---|" * len(horizon_ids)]
            rows = [
                ("平均 Rank IC", [s["rank_ic"]["mean"] for s in summaries]),
                ("固定方向平均 Rank IC", [None if s["rank_ic"]["mean"] is None else s["rank_ic"]["mean"] * horizons[str(h)]["direction"]
                                         for h, s in zip(horizon_ids, summaries)]),
                ("Rank IC 标准误", [s["rank_ic"]["se"] for s in summaries]),
                ("Rank IC 置信区间", [s["rank_ic"]["ci"] for s in summaries]),
                ("原始 p 值", [tests[h]["raw_p"] for h in horizon_ids]),
                ("BH 调整后 p 值", [tests[h]["adjusted_p"] for h in horizon_ids]),
                ("BH 检验通过", ["是" if tests[h]["rejected"] else "否" for h in horizon_ids]),
                ("ICIR（未年化）", [s["rank_ic"]["mean_std_ratio"] for s in summaries]),
                ("有向高低组收益差", [s["directional_spread"]["mean"] for s in summaries]),
                ("价差置信区间", [s["directional_spread"]["ci"] for s in summaries]),
                ("有效 IC 期数", [s["rank_ic"]["n"] for s in summaries]),
                ("有效资产小时观测", [horizons[str(h)]["coverage"]["eligible_observations"]
                                    for h in horizon_ids]),
                ("跨段排除资产小时观测", [horizons[str(h)]["coverage"]["purged_observations"]
                                      for h in horizon_ids]),
                ("跨段排除小时数", [horizons[str(h)]["coverage"]["purged_hours"]
                                    for h in horizon_ids]),
                ("逐期限程序判定", [horizon_results[str(h)]["validation_status"] for h in horizon_ids]),
                ("逐期限Plan3轨道", ["、".join(horizon_results[str(h)]["tracks"]) or "无"
                                   for h in horizon_ids]),
            ]
            rows += [(f"第{group}组平均收益", [s["group_means"][group] for s in summaries])
                     for group in summaries[0]["group_means"]]
            for label, values in rows:
                lines.append("| " + label + " | " + " | ".join(_cell(value) for value in values) + " |")
            lines += ["", "逐期限分组收益图：" + "、".join(
                f"[{h}h](plots/{cid}-B-{h}h.svg)" for h in horizon_ids), "",
                f"整批程序判定：{'通过' if outcome['validation_status'] == 'passed' else '未通过'}；"
                f"创意卡交付资格：{'满足' if outcome['eligible_for_idea_pool'] else '不满足'}。",
                "A保留期限：" + "、".join(f"{h}h" for h in outcome["retained_horizons"]),
                "B通过期限：" + "、".join(f"{h}h" for h in outcome["passed_horizons"]),
                f"准入规则：{outcome['admission_scheme']}；Plan3轨道标签：" + "、".join(outcome["tracks"]), ""]
            lines += [f"- {reason}" for reason in outcome["reasons"]]
            for horizon in horizon_ids:
                lines += [f"- {horizon}h：" + ("；".join(horizon_results[str(horizon)]["reasons"])
                          or "无未通过原因")]
            lines += ["", f"[完整程序判定](b_records/{validations[cid]['id']}.json)。模型解释不改变此结果。", ""]
        _rank_displacement_section(lines, path.parent, data.get("rank_displacement"),
                                   data.get("rank_displacement_archive"))
        if cid in reports:
            model = reports[cid]
            lines += ["### 模型解释", "", model["analysis"], "", "适用及失效条件：", ""]
            lines += [f"- {value}" for value in model["conditions"]]
            lines += ["", "证据限制：", ""] + [f"- {value}" for value in model["limitations"]] + [""]
        archive_refs = ([(primary_horizon, data["factor_archive"])] if "factor_archive" in data else []) if stage == "A" else [
            (horizon, horizons[str(horizon)]["factor_archive"]) for horizon in horizon_ids]
        for horizon, pointer in archive_refs:
            archive_path = _archive_file(path.parent, pointer)
            relative_archive = Path(os.path.relpath(archive_path, path.parent.resolve())).as_posix()
            evaluation_id = pointer["evaluation_id"]
            value_set_id = pointer.get("value_set_id")
            value_set_note = f" · factor value set `{value_set_id}`" if value_set_id is not None else ""
            lines += [f"稳定定位：运行 `{run_id}` · 候选 `{cid}` · SQLite evaluation `{evaluation_id}` · {horizon}h{value_set_note}。",
                      f"精确归档位置：[SQLite 因子归档]({relative_archive})。该 JSON 只含统计摘要和覆盖范围；"
                      "逐期、分阶段、分组及因子值证据保存在归档中。", ""]
        record_path = f"{stage.lower()}_records/{record['id']}.json"
        lines += [f"保存的逐期限评估记录：[{record['id']}.json]({record_path})。", ""]
    unsuccessful = [r for r in records if r["kind"] == "duplicate"
                    or (r["kind"] == "calculation" and r["data"]["status"] != "computed")]
    if unsuccessful:
        lines += ["## 失败、重复与待补充定义", ""]
        for record in unsuccessful:
            data = record["data"]
            lines.append(f"- {data['candidate_id']}：{data.get('status', record['kind'])}，见 `{stage.lower()}_records/{record['id']}.json`。")
    with path.open("w" if replace else "x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
