#!/usr/bin/env python3
"""A-only IC calibration and saved-B policy comparison for the fixed FM-v6 Goal."""
from __future__ import annotations

import argparse
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from compare_fm_v6_horizons import (
    DEFAULT_SOURCE_GOAL, HORIZONS, archive_map_for_group, canonical_json, factor_key, file_signature, load_archive_identity,
    load_archive_payload, read_json, require, validate_a_payload, verify_signature,
    validate_b_archive, write_new_json,
)

THRESHOLDS = (0.02, 0.015, 0.01)


def quantiles(values):
    if not values:
        return None
    values = sorted(values)
    def percentile(q):
        position = (len(values) - 1) * q
        lo = int(position)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (position - lo)
    return {str(q): percentile(q) for q in (0, .1, .25, .5, .75, .9, 1)}


def directed_interval(ci, direction):
    return None if ci is None else sorted(direction * value for value in ci)


def ratio(values, predicate):
    return sum(predicate(value) for value in values) / len(values) if values else None


def diagnose(report, direction):
    summary = report['summary']
    ic = summary['rank_ic']
    stages = [{
        'start': row['start'], 'end': row['end'],
        'directed_ic': None if row['rank_ic']['mean'] is None else direction * row['rank_ic']['mean'],
        'directed_ci': directed_interval(row['rank_ic']['ci'], direction),
        'icir': None if row['rank_ic']['mean_std_ratio'] is None else direction * row['rank_ic']['mean_std_ratio'],
        'n': row['rank_ic']['n'], 'directional_spread': row['directional_spread']['mean'],
    } for row in report['stages']]
    stage_ics = [row['directed_ic'] for row in stages if row['directed_ic'] is not None]
    years = {}
    for row in report['periods']:
        if row['rank_ic'] is not None:
            years.setdefault(row['timestamp'][:4], []).append(direction * row['rank_ic'])
    symbols = report['per_symbol']
    total_symbol_n = sum(row['n'] for row in symbols)
    groups = summary['group_means']
    means = [groups[key] for key in sorted(groups, key=int)]
    monotone_pairs = [direction * (right - left) >= 0 for left, right in zip(means, means[1:])
                      if left is not None and right is not None]
    return {
        'directed_ic': None if ic['mean'] is None else direction * ic['mean'],
        'directed_ci': directed_interval(ic['ci'], direction),
        'icir': None if ic['mean_std_ratio'] is None else direction * ic['mean_std_ratio'],
        'rank_ic_statistics': ic, 'directional_spread': summary['directional_spread'],
        'group_means': groups, 'adjacent_groups_in_direction': sum(monotone_pairs),
        'adjacent_groups_estimable': len(monotone_pairs),
        'coverage': report['coverage'], 'stage_count': len(stages),
        'valid_stage_count': len(stage_ics), 'positive_ic_stage_share': ratio(stage_ics, lambda v: v > 0),
        'stage_ic_quantiles': quantiles(stage_ics), 'stages': stages,
        'year_ic': {year: {'n': len(values), 'directed_ic': statistics.mean(values)}
                    for year, values in sorted(years.items())},
        'per_symbol': symbols,
        'largest_symbol_observation_share': max((row['n'] / total_symbol_n for row in symbols), default=None)
            if total_symbol_n else None,
        'asset_predictive_contribution': 'not estimable from archived per_symbol aggregates; these contain coverage and means, not per-asset IC influence',
    }


def calibrate(args):
    source = args.source_goal.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_new_json(output / 'analysis_scope.json', {
        'created_at': datetime.now(timezone.utc).isoformat(), 'source_goal': str(source),
        'thresholds': list(THRESHOLDS), 'expected_unique_factors': 44,
        'horizons': list(HORIZONS), 'selection_inputs': 'A only; all retain/pause/discard candidates',
        'threshold_selection': 'human-readable research judgment after A analysis; no automatic optimum',
        'prior_B_inspection': True, 'B_read_during_calibration': False,
        'family_grouping': 'original exploration cycle; descriptive themes, not independent statistical clusters',
    })
    archives = {}
    for path in sorted((source / 'runs/factor_archive_v2').glob('factor-*.sqlite3')):
        key = factor_key(load_archive_identity(path))
        require(key not in archives, f'duplicate factor archive {key}')
        archives[key] = path
    candidates = {}
    for batch, run in enumerate(sorted((source / 'runs').glob('goal-*')), 1):
        decisions = read_json(run / 'a-complete.json')['candidate_decisions']
        for path in sorted((run / 'a_records').glob('*-calculation.json')):
            calculation = read_json(path)['data']
            cid = calculation['candidate_id']
            definition_path = run / 'a_records' / f'{cid}-definition.json'
            evaluation_path = run / 'a_records' / f'{cid}-evaluation.json'
            definition = read_json(definition_path)['data']['definition']
            evaluation = read_json(evaluation_path)['data']
            locator = evaluation['factor_archive']
            identity = locator['identity']
            key = factor_key(identity)
            require(identity['expanded_expression'] == calculation['executed_expression']['expanded_expression']
                    and int(identity['direction']) == definition['direction'], f'identity differs: {path}')
            archive = archives[key]
            signature = file_signature(archive)
            payload, row = load_archive_payload(archive, locator['evaluation_id'])
            require(locator['evaluation_key'] == {field: row[field] for field in
                    ('data_version', 'contract_version', 'evaluator_version', 'segment', 'horizon')},
                    f'A locator key differs: {evaluation_path}')
            validate_a_payload(payload, identity, key)
            verify_signature(signature, 'A archive')
            occurrence = {'source_run': str(run), 'candidate_id': cid, 'batch': batch,
                          'original_disposition': decisions[cid]['disposition'],
                          'definition_path': str(definition_path), 'calculation_path': str(path),
                          'evaluation_path': str(evaluation_path), 'archive': str(archive),
                          'archive_signature': signature, 'evaluation_id': locator['evaluation_id'],
                          'evaluation_key': locator['evaluation_key']}
            if key in candidates:
                candidates[key]['occurrences'].append(occurrence)
                continue
            candidates[key] = {
                'identity': identity, 'name': definition['name'], 'family_cycle': batch,
                'original_disposition': decisions[cid]['disposition'], 'occurrences': [occurrence],
                'horizons': {str(h): diagnose(payload['horizon_comparison']['horizons'][str(h)],
                                             int(identity['direction'])) for h in HORIZONS},
            }
    require(len(candidates) == 44, f'expected full 44-factor inventory, got {len(candidates)}')
    summary = {}
    for h in map(str, HORIZONS):
        valid = {key: item['horizons'][h] for key, item in candidates.items()
                 if item['horizons'][h]['directed_ic'] is not None}
        thresholds = {}
        for threshold in THRESHOLDS:
            selected = sorted(key for key, item in valid.items() if item['directed_ic'] >= threshold)
            weak = sorted(key for key in selected if valid[key]['directed_ic'] < .02)
            thresholds[str(threshold)] = {
                'factor_keys': selected, 'count': len(selected), 'below_002_keys': weak,
                'below_002_count': len(weak),
                'below_002_families': dict(Counter(candidates[key]['family_cycle'] for key in weak)),
                'weak_positive_stage_share': quantiles([valid[key]['positive_ic_stage_share'] for key in weak
                                                         if valid[key]['positive_ic_stage_share'] is not None]),
                'weak_positive_all_years': [key for key in weak if all(
                    item['directed_ic'] > 0 for item in valid[key]['year_ic'].values())],
                'ic_spread_disagreement': [key for key in selected if valid[key]['directional_spread']['mean']
                                          is not None and valid[key]['directional_spread']['mean'] <= 0],
            }
        summary[h] = {'total': len(candidates), 'estimable': len(valid),
                      'missing_keys': sorted(set(candidates) - set(valid)),
                      'directed_ic_quantiles': quantiles([item['directed_ic'] for item in valid.values()]),
                      'positive': sum(item['directed_ic'] > 0 for item in valid.values()),
                      'negative': sum(item['directed_ic'] < 0 for item in valid.values()),
                      'zero': sum(item['directed_ic'] == 0 for item in valid.values()),
                      'thresholds': thresholds}
    result = {'source_goal': str(source), 'factor_count': len(candidates),
              'dispositions': dict(Counter(item['original_disposition'] for item in candidates.values())),
              'summary': summary, 'candidates': candidates}
    write_new_json(output / 'a_calibration.json', result)
    lines = ['# FM-v6 全样本 A 段 IC 标定', '',
             '固定比较 0.02、0.015、0.01；全部 44 个公式与方向去重后纳入，A 不增加硬门槛。', '',
             '| 期限 | 可估计/全部 | 正/负/零 | IC 中位数 | ≥0.02 | ≥0.015 | ≥0.01 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for h, item in summary.items():
        lines.append(f"| {h}h | {item['estimable']}/44 | {item['positive']}/{item['negative']}/{item['zero']} | "
                     f"{item['directed_ic_quantiles']['0.5']:.6f} | " + ' | '.join(
                         str(item['thresholds'][str(t)]['count']) for t in THRESHOLDS) + ' |')
    lines += ['', '## 相对 0.02 的弱信号', '',
              '| 期限 | T | 新覆盖 A 线索 | 原周期族数量 | 正 IC 阶段占比中位数 | 全部年份同向数 | IC/价差矛盾数 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for h, item in summary.items():
        for t in (.015, .01):
            row = item['thresholds'][str(t)]
            q = row['weak_positive_stage_share']
            lines.append(f"| {h}h | {t} | {row['below_002_count']} | {len(row['below_002_families'])} | "
                         f"{q['0.5'] if q else None} | {len(row['weak_positive_all_years'])} | "
                         f"{len(row['ic_spread_disagreement'])} |")
    lines += ['', '## 逐因子证据', '',
              '| 身份 | 原去向 | 原周期族 | 期限 | 有向 IC | 有向 CI | ICIR | 正 IC 阶段占比 | 价差 | 年份 IC |',
              '|---|---|---:|---:|---:|---|---:|---:|---:|---|']
    for key, candidate in candidates.items():
        for h, row in candidate['horizons'].items():
            lines.append(f"| {key} | {candidate['original_disposition']} | {candidate['family_cycle']} | {h}h | "
                         f"{row['directed_ic']} | {row['directed_ci']} | {row['icir']} | "
                         f"{row['positive_ic_stage_share']} | {row['directional_spread']['mean']} | "
                         f"{row['year_ic']} |")
    lines += ['', '完整 168h 阶段的 CI、ICIR、样本数、分组收益和币种覆盖见 a_calibration.json。',
              '原周期族仅为主题诊断分组，相似变体不能视作独立重复证据。',
              '归档逐币数据只有覆盖与均值，不能识别单币对横截面 IC 的贡献；币种观测份额只说明覆盖集中程度。',
              '旧 B 已被查看；本标定只读取 A 归档，A 曾参与自适应开发，后续独立验证仍待确定。', '']
    (output / 'a_calibration.md').write_text('\n'.join(lines), encoding='utf-8')
    print({h: {t: row['count'] for t, row in item['thresholds'].items()} for h, item in summary.items()})


def compare(args):
    from dataclasses import replace
    from crypto_quant.research.factor_mining.contracts import ResearchSpec
    from crypto_quant.research.factor_mining.workflow import b_candidate_decision
    output = args.output.resolve(strict=True)
    decision = read_json(output / 'threshold_decision.json')
    selected = decision['selected_threshold']
    require(selected in THRESHOLDS, 'threshold must be one of the registered candidates')
    spec = ResearchSpec.from_dict(read_json(output / 'research.contract.json'))
    require(spec.min_abs_ic == selected, 'decision and new contract threshold differ')
    source = args.comparison.resolve(strict=True)
    report = read_json(source)
    results, sets, evidence = {}, {str(t): [] for t in THRESHOLDS}, []
    require(len(report['factor_results']) == 26, 'saved B comparison must cover 26 identities')
    run_dirs = sorted({Path(row['new_multi']['run_path']) for row in report['factor_results'].values()})
    archive_root, archives = archive_map_for_group(run_dirs[0].parent.parent, run_dirs)
    for key, candidate in report['factor_results'].items():
        prior = candidate['new_multi']
        evaluation_path = Path(prior['evaluation_record_path'])
        validation_path = Path(prior['validation_record_path'])
        run = Path(prior['run_path'])
        correction_path = run / 'b_records/batch-correction.json'
        evaluation = read_json(evaluation_path)['data']
        validation = read_json(validation_path)['data']
        correction = read_json(correction_path)['data']
        old_contract = read_json(run / 'contract.json')
        old_spec = ResearchSpec.from_dict(old_contract)
        require(old_spec.min_abs_ic == .02, 'saved B baseline IC differs')
        informational = {'run_id', 'objective', 'data_usage_review', 'min_abs_ic'}
        require({k: v for k, v in old_contract.items() if k not in informational} ==
                {k: v for k, v in spec.as_dict().items() if k not in informational},
                'new policy changed a calculation or statistical setting')
        frozen_path = run / 'frozen_batch.json'
        frozen = read_json(frozen_path)['candidates'][candidate['candidate_id']]
        identity = frozen['a_evaluation_archive']['identity']
        require(factor_key(identity) == key and
                identity['expanded_expression'] == candidate['expanded_expression'] and
                int(identity['direction']) == int(candidate['direction']), 'B frozen formula identity differs')
        require(validation['candidate_id'] == candidate['candidate_id'] == evaluation['candidate_id'], 'B identity mismatch')
        require(evaluation['retained_horizons'] == prior['retained_horizons'], 'frozen horizons mismatch')
        require(int(evaluation['direction']) == int(candidate['direction']), 'B direction mismatch')
        evidence.append(file_signature(archives[canonical_json(identity)]))
        evaluation['horizons'] = {h: validate_b_archive(
            run, archive_root, archives, compact['factor_archive'], identity, int(h), compact)
            for h, compact in evaluation['horizons'].items()}
        baseline_decision = b_candidate_decision(candidate['candidate_id'], evaluation, correction, old_spec)
        require(baseline_decision == {field: validation[field] for field in baseline_decision},
                'saved B baseline decision does not reproduce')
        # All calculation/statistical settings remain the old ones; only this effect threshold changes.
        policies = {}
        for threshold in THRESHOLDS:
            policies[str(threshold)] = b_candidate_decision(candidate['candidate_id'], evaluation,
                                                           correction, replace(old_spec, min_abs_ic=threshold))
            if policies[str(threshold)]['eligible_for_idea_pool']:
                sets[str(threshold)].append(key)
        results[key] = {'candidate_id': candidate['candidate_id'], 'source_run': str(run),
                        'direction': candidate['direction'], 'expanded_expression': candidate['expanded_expression'],
                        'frozen_horizons': prior['retained_horizons'], 'policies': policies,
                        'evaluation_path': str(evaluation_path), 'validation_path': str(validation_path),
                        'correction_path': str(correction_path), 'calculation_contract_path': str(run / 'contract.json')}
        evidence.extend(file_signature(path, hash_contents=True) for path in
                        (evaluation_path, validation_path, correction_path, run / 'contract.json', frozen_path))
    sets = {t: sorted(keys) for t, keys in sets.items()}
    baseline = set(sets['0.02'])
    require(len(baseline) == 5 and baseline == {key for key, row in report['factor_results'].items()
                                               if row['new_multi']['qualified']}, 'baseline 5 identities differ')
    changes = {}
    for t, keys in sets.items():
        current = set(keys)
        require(baseline <= current, 'lower IC threshold unexpectedly loses a baseline identity')
        changes[t] = {'numeric_qualified_count': len(current), 'retained': sorted(current & baseline),
                      'added': sorted(current - baseline), 'lost': sorted(baseline - current),
                      'net_change': len(current) - len(baseline)}
    for signature in evidence:
        verify_signature(signature, 'saved B evidence')
    write_new_json(output / 'b_policy_comparison.json', {
        'source': str(source), 'source_signature': file_signature(source, hash_contents=True),
        'selected_threshold': selected, 'identity_sets': sets, 'changes': changes,
        'factor_results': results, 'input_signatures': evidence, 'actual_new_cards_written': 0,
        'evidence_type': 'post-selection historical numeric policy reassessment; no B prices or model calls',
    })
    lines = ['# FM-v6 保存 B 数值的 IC 规则影响对照', '',
             f'选定 T={selected}。固定 26 个身份、62 项候选×期限检验和原 7 个批次的 BH 家族。', '',
             '| T | 数值合格因子 | 保留 | 新增 | 流失 | 净增 |', '|---:|---:|---:|---:|---:|---:|']
    for t, change in changes.items():
        lines.append(f"| {t} | {change['numeric_qualified_count']} | {len(change['retained'])} | "
                     f"{len(change['added'])} | {len(change['lost'])} | {change['net_change']} |")
    lines += ['', '| 身份 | 原通过期限 | 新通过期限 | 变化原因 |', '|---|---|---|---|']
    for key, row in results.items():
        old, new = row['policies']['0.02'], row['policies'][str(selected)]
        reason = 'IC 下限降低，其余判定和送检期限不变' if old['passed_horizons'] != new['passed_horizons'] else '无变化'
        lines.append(f"| {key} | {old['passed_horizons']} | {new['passed_horizons']} | {reason} |")
    lines += ['', '本次实际新增交付卡片为 0。数值合格的新增身份须经独立历史再验证运行补齐报告、凭据和 Goal 匹配后才可写卡。',
              '旧合同、验证记录、卡片、Goal 收据及全局创意池保持原状。新门槛尚未在未参与选择的数据上验证。', '']
    (output / 'b_policy_comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    print(changes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    a = commands.add_parser('calibrate')
    a.add_argument('--source-goal', type=Path, default=DEFAULT_SOURCE_GOAL)
    a.add_argument('--output', type=Path, required=True)
    a.set_defaults(run=calibrate)
    b = commands.add_parser('compare')
    b.add_argument('--comparison', type=Path, required=True)
    b.add_argument('--output', type=Path, required=True)
    b.set_defaults(run=compare)
    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
