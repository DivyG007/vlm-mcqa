"""Independent CPU-only audit of the frozen pkgpu LLaVA Core-7 campaign.

Reads exported evidence without regenerating or rewriting experiment outputs.
Checks hashes, JSONL coverage/counts, raw-derived means and replication checks.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9), (a, b)


def pair_means(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['pair_id']].append(row['normalized_recovery'])
    return mean(mean(values) for values in groups.values())


def audit(campaign):
    workflow = read(campaign / 'campaign.json')
    assert workflow['status'] == 'complete'
    assert len(workflow['stages']) == 5 and all(s['exit_code'] == 0 for s in workflow['stages'])
    base = campaign / 'llava_core7'
    frozen = read(base / 'freeze.json')
    payload = {k: v for k, v in frozen.items() if k != 'freeze_hash'}
    assert hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16] == frozen['freeze_hash']
    output = {'campaign': workflow, 'freeze_hash': frozen['freeze_hash'], 'phases': {}}
    phase_pairs = {}
    for split in ('discovery', 'confirmation'):
        folder = base / split
        run, cal = read(folder / 'run.json'), read(folder / 'calibration.json')
        assert run['status'] == 'complete' and cal['all_passed']
        assert run.get('calibration_passed', True)
        assert len(cal['gates']) == 9 and all(g['passed'] for g in cal['gates'].values())
        for key in ('source_git_commit', 'dataset_hash', 'image_aware_release_hash', 'model_key'):
            assert run[key] == frozen[key], key
        assert run['source_tree_dirty'] is False and run['profile_name'] == 'full_core7'
        assert run['model_spec']['revision'] == 'cae7503dce88887557f73db1882328d21d9a7e21'
        assert run['profile']['pairs'] == 64 and run['profile']['batch_size'] == 4
        assert run['adapter']['dtype'] == 'bfloat16'
        rows_by_file, counts = {}, {}
        manifest = read(folder / 'raw_sha256.json')
        for name, expected in manifest.items():
            assert sha(folder / name) == expected, name
        for path in sorted(folder.glob('*.jsonl')):
            with path.open(encoding='utf-8') as stream:
                rows = [json.loads(line) for line in stream]
            assert rows, path.name
            assert path.name in manifest, path.name
            counts[path.name] = len(rows)
            rows_by_file[path.name] = rows
            for row in rows:
                for metric in ('normalized_recovery', 'margin_drop', 'margin'):
                    if metric in row:
                        assert math.isfinite(row[metric]), (path.name, metric)
        patch = read(folder / f'patching_{split}_summary.json')
        comps = read(folder / f'components_{split}_summary.json')
        heads = read(folder / f'heads_{split}_summary.json')
        aggregate = read(folder / f'aggregate_{split}.json')
        assert counts[f'patching_{split}.jsonl'] == patch['m1_rows'] + patch['group_patching']['rows']
        assert counts[f'components_{split}.jsonl'] == comps['patch_rows']
        assert counts[f'component_ablation_{split}.jsonl'] == comps['ablation_rows'] == 128 * 28 * 2
        assert counts[f'heads_ablation_{split}.jsonl'] == heads['ablation_rows'] == 128 * 11
        assert heads['candidates'] == frozen['head_candidates'] and heads['topk'] == frozen['head_set_sizes']
        assert heads['cases']['limited_case_selection'] == 'round_robin_across_pairs'
        assert heads['cases']['selected_pairs_by_type']['position'] == 64
        n_cases = heads['cases']['cases_by_type']['position']
        assert n_cases == 96
        assert counts[f'heads_validate_{split}.jsonl'] == n_cases * (16 + len(heads['topk']) * 21 + 1)
        if split == 'discovery':
            assert counts['heads_screen_discovery.jsonl'] == n_cases * len(heads['head_layers']) * run['adapter']['num_heads']
            lens = read(folder / 'lens_behavior_summary.json')
            assert counts['lens_behavior.jsonl'] == lens['stage_rows'] == lens['variants'] * 29
            assert counts['component_projection_behavior.jsonl'] == lens['component_rows'] == lens['variants'] * 28 * 2
            assert counts['compliance_behavior.jsonl'] == 256
            behavior = read(folder / 'behavior_behavior_summary.json')
            behavior_rows = rows_by_file['behavior_behavior.jsonl']
            clean = [r for r in behavior_rows if r['split'] == 'behavior' and r['condition'] == 'clean']
            assert len({r['item_id'] for r in clean}) == behavior['items'] == 256
            letters = [r for r in clean if r['scheme'] == 'letters']
            close(mean(r['correct'] for r in letters), behavior['letters_accuracy'])
            for condition in ('blank', 'shuffled'):
                control = [r for r in behavior_rows if r['split'] == 'behavior' and r['condition'] == condition]
                close(mean(r['correct'] for r in control), behavior[condition + '_accuracy'])
            close(behavior['letters_accuracy'] - behavior['shuffled_accuracy'], behavior['clean_minus_shuffled'])
        patch_rows = rows_by_file[f'patching_{split}.jsonl']
        assert len({(r['contrast_id'], r['method'], r['layer'], r['group']) for r in patch_rows}) == len(patch_rows)
        phase_pairs[split] = {r['pair_id'] for r in patch_rows}
        assert len(phase_pairs[split]) == 64
        assert patch['m1_rows'] == sum(patch['cases']['cases_by_type'].values()) * 28
        assert comps['patch_rows'] == sum(comps['cases']['cases_by_type'].values()) * 28 * 2
        for contrast in ('position', 'content'):
            grouped = defaultdict(list)
            for row in patch_rows:
                if row['group'] == 'final' and row['contrast_type'] == contrast:
                    grouped[row['layer']].append(row['normalized_recovery'])
            curve = {layer: mean(values) for layer, values in grouped.items()}
            for layer, value in curve.items():
                close(value, aggregate[contrast + '_curve'][str(layer)])
            assert next((l for l in sorted(curve) if curve[l] >= .5), None) == aggregate[contrast + '_transfer_layer']
        validation = rows_by_file[f'heads_validate_{split}.jsonl']
        by_set = defaultdict(list)
        for row in validation:
            by_set[row['set']].append(row)
        assert len({(r['contrast_id'], r['set']) for r in validation}) == len(validation)
        close(pair_means(by_set['all_heads']), heads['all_heads_mean_nr'])
        for k in heads['topk']:
            saved = heads['per_k'][str(k)]
            close(pair_means(by_set[f'top{k}']), saved['top_k']['mean'])
            random = sorted(pair_means(by_set[f'random{k}_{r}']) for r in range(20))
            close(random[18], saved['random_p95'])
            close(mean(random), saved['random_mean'])
            assert saved['top_k']['n_clusters'] == 64 and saved['top_k']['sampling_unit'] == 'pair'
            assert saved['beats_random_p95'] == (saved['top_k']['mean'] > saved['random_p95'])
        small = any(k <= 4 and heads['per_k'][str(k)]['beats_random_p95'] and
                    heads['per_k'][str(k)]['fraction_of_all_heads'] >= .5 for k in heads['topk'])
        assert heads['verdict'][0] == ('sparse' if small else 'distributed')
        from PIL import Image
        figures = sorted(folder.glob('figures/*.png'))
        assert len(figures) == (10 if split == 'discovery' else 6)
        for figure in figures:
            with Image.open(figure) as im:
                im.verify()
        output['phases'][split] = {'rows': counts, 'raw_hashes_verified': len(manifest),
                                  'figures': len(figures), 'calibration_gates': len(cal['gates']),
                                  'pair_count': len(phase_pairs[split]),
                                  'method_seconds': run['method_seconds'],
                                  'wall_seconds': run['finished_unix'] - run['started_unix']}
    assert not phase_pairs['discovery'] & phase_pairs['confirmation']
    report = read(base / 'confirmation/confirmation_report.json')
    assert report['freeze_hash'] == frozen['freeze_hash']
    assert report['checks']['execution_complete'] is True
    output.update(checks=report['checks'], disjoint_pair_splits=True, passed=True)
    logs = {}
    for path in campaign.glob('*.log'):
        content = path.read_text(encoding='utf-8', errors='replace')
        fatal = [pattern for pattern in ('Traceback (most recent call last)', 'CUDA out of memory',
                                         'calibration failed:', '[stop]') if pattern in content]
        assert not fatal, (path.name, fatal)
        logs[path.name] = {'fatal_patterns': fatal, 'bytes': path.stat().st_size}
    output['logs'] = logs
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.campaign)
    args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
