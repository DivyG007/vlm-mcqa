"""Offline, reproducible audit of the four preserved IconQA Core-7 runs.

Reads immutable backups; writes a new analysis package, never run artifacts.
Usage: python scripts/local/analyze_iconqa_transfer_runs.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
BACKUPS = ROOT / 'results/ada_completed_run_backups'
OUT = ROOT / 'consolidation/iconqa_transfer_analysis_20261030'
JOBS = ['45569', '45585', '46804', '46912']
SMALL = ['46804', '46912']
NAMES = {'45569': 'Qwen natural B1 (cancelled)', '45585': 'Qwen natural B4',
         '46804': 'Qwen CF-small B4', '46912': 'InternVL CF-small B4'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    if path.exists():
        with path.open(encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def mean(values):
    a = [float(x) for x in values if math.isfinite(float(x))]
    return sum(a) / len(a) if a else None


def interval(values):
    a = np.array(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {'mean': None, 'ci_low': None, 'ci_high': None, 'source_units': 0}
    rng = np.random.default_rng(20261030)
    draws = a[rng.integers(0, len(a), (10000, len(a)))].mean(axis=1)
    low, high = np.quantile(draws, [.025, .975])
    return {'mean': float(a.mean()), 'ci_low': float(low), 'ci_high': float(high),
            'source_units': len(a)}


def clustered(records, value, cluster='pair_id'):
    by = defaultdict(list)
    for r in records:
        x = r.get(value)
        if x is not None and math.isfinite(float(x)):
            by[r[cluster]].append(float(x))
    return interval([mean(v) for _, v in sorted(by.items())])


def write_csv(name, records):
    if not records:
        return
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, sorted(set().union(*(r.keys() for r in records))))
        w.writeheader()
        w.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-summary-only', action='store_true',
                        help='Export compact publication files from hash-verified existing metrics.')
    args = parser.parse_args()
    if args.export_summary_only:
        manifest = read(OUT / 'analysis_manifest.json')
        assert sha(OUT / 'metrics.json') == manifest['outputs']['metrics.json']
        publish_summary(read(OUT / 'metrics.json'))
        print('Compact summary and publication manifest exported.', flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'figures').mkdir(exist_ok=True)
    metrics = {'method': 'Source-pair means, percentile bootstrap: 10000 draws, seed 20261030. '
               'Source-unit intervals are descriptive; no multiple-comparison correction.',
               'runs': {}, 'comparison_policy': 'Only CF-small runs enter final model comparison.'}
    curve_rows, behavior_rows, lens_rows, control_rows = [], [], [], []
    behavior_raw = {}
    for job in JOBS:
        p = BACKUPS / ('job_' + job)
        inv = read(p / 'remote_inventory.json')
        receipt = read(p / 'transfer_complete.json')
        assert receipt['transfer_status'] == 'verified' and receipt['transfer_complete']
        assert receipt['inventory_sha256'] == sha(p / 'remote_inventory.json')
        failures = []
        for name, entry in inv['files'].items():
            local = Path(entry['local_path']) if 'local_path' in entry else p / name
            if not local.is_file() or local.stat().st_size != entry['bytes'] or sha(local) != entry['sha256']:
                failures.append(name)
        assert not failures, (job, failures)
        print('Verified all inventory files:', job, len(inv['files']), flush=True)
        result = p / 'results'
        frozen = read(result / 'iconqa_freeze.json')
        ri = read(result / 'resume_identity.json')
        run = read(result / ('run.json' if (result / 'run.json').exists() else 'discovery/run.json'))
        out = {'name': NAMES[job], 'backup': str(p), 'source_commit': inv['source_commit'],
               'run_state': inv['run_state'], 'accounting': inv.get('sacct', inv.get('accounting')),
               'verified_inventory_files': len(inv['files']), 'freeze': frozen,
               'resume_identity': ri, 'profile': run.get('profile'), 'phases': {}}
        metrics['runs'][job] = out
        behavior_raw[job] = {}
        for phase in ['discovery', 'confirmation']:
            q = result / phase
            b = read(q / f'behavior_{phase}_summary.json')
            br = list(rows(q / f'behavior_{phase}.jsonl'))
            behavior_raw[job][phase] = br
            r = {'behavior': b, 'eligibility': read(q / 'eligibility.json'),
                 'calibration_all_passed': read(q / 'calibration.json')['all_passed'],
                 'checkpoints': sorted(x.stem for x in (q / '.checkpoints').glob('*.json')),
                 'raw_counts': {}, 'raw_nonfinite': {}, 'checkpoint_artifacts_verified': 0}
            for cp in (q / '.checkpoints').glob('*.json'):
                for remote, digest in read(cp)['artifacts'].items():
                    match = [v for v in inv['files'].values() if v['remote_path'] == remote]
                    assert match and any(v['sha256'] == digest for v in match), (job, cp, remote)
                    r['checkpoint_artifacts_verified'] += 1
            for f in q.glob('*.jsonl'):
                counts = 0
                bad = Counter()
                for record in rows(f):
                    counts += 1
                    for k in ['normalized_recovery', 'margin', 'patched_diff']:
                        if k in record and not math.isfinite(float(record[k])):
                            bad[k] += 1
                    if 'normalized_recovery' in record:
                        denominator = record['donor_diff'] - record['recipient_diff']
                        expected_nr = (record['patched_diff'] - record['recipient_diff']) / denominator
                        assert math.isclose(record['normalized_recovery'], expected_nr, rel_tol=1e-10, abs_tol=1e-10), (job,f,counts)
                r['raw_counts'][f.stem] = counts
                if bad:
                    r['raw_nonfinite'][f.stem] = dict(bad)
            agg = q / f'aggregate_{phase}.json'
            if agg.exists():
                r['aggregate'] = read(agg)
            cf = q / 'clevr_frozen_heads' / f'heads_{phase}_summary.json'
            if cf.exists():
                r['clevr_heads'] = read(cf)
                r['clevr_source_freeze'] = read(cf.parent / 'source_freeze.json')
            cr = q / 'confirmation_report.json'
            if cr.exists():
                r['replication_checks'] = read(cr)['checks']
            out['phases'][phase] = r
            if job in SMALL:
                # Full behavior population: 64 source pairs, two images, four layouts.
                r['behavior_strata'] = {}
                dataset_entries = {n:v for n,v in inv['files'].items() if n == 'dataset/items.jsonl'}
                entry = dataset_entries['dataset/items.jsonl']
                items_path = Path(entry['local_path']) if 'local_path' in entry else p / 'dataset/items.jsonl'
                items = {i['item_id']: i for i in rows(items_path)}
                clean = [x for x in br if x['condition'] == 'clean']
                for x in clean:
                    x.update(image_status=items[x['item_id']]['image_status'],
                             semantic_answer=items[x['item_id']]['semantic_answer'])
                for scheme in ['letters', 'numbers', 'rare_letters']:
                    ss = [x for x in clean if x['scheme'] == scheme]
                    for group, subset in [('all', ss)] + [(status, [x for x in ss if x['image_status'] == status]) for status in ['native', 'edited']]:
                        stat = clustered(subset, 'correct')
                        stat.update(variant_records=len(subset), item_count=len({x['item_id'] for x in subset}))
                        r['behavior_strata'][scheme + '/' + group] = stat
                        behavior_rows.append(dict(job=job, phase=phase, scheme=scheme, stratum=group, **stat))
                    per_class = {}
                    for cls in ['certain', 'probable', 'impossible', 'unlikely']:
                        subset = [x for x in ss if x['semantic_answer'] == cls]
                        per_class[cls] = dict(accuracy=mean(x['correct'] for x in subset), variants=len(subset))
                    r['behavior_strata'][scheme + '/classes'] = per_class
                    r['behavior_strata'][scheme + '/macro_accuracy'] = mean(x['accuracy'] for x in per_class.values())
                selected = set(r['eligibility']['pair_ids'])
                r['behavior_selected_pairs'] = sorted(selected)
                controls = list(rows(q / f'edit_controls_{phase}.jsonl'))
                cstat = {}
                for condition in ['counterfactual', 'same_answer']:
                    sub = [dict(pair_id=x['pair_id'], **x['conditions'][condition]) for x in controls]
                    cstat[condition] = {key: clustered(sub, key) for key in ['patched_minus_native', 'clean_diff', 'patched_diff']}
                    # A positive two-answer margin is not a verified four-way flip.
                    cstat[condition]['positive_two_answer_margin_fraction'] = mean(x['patched_diff'] > 0 for x in sub)
                differences = [x['conditions']['counterfactual']['patched_minus_native'] - x['conditions']['same_answer']['patched_minus_native'] for x in controls]
                cstat['paired_answer_change_minus_same_answer'] = interval(differences)
                cstat['frozen_heads_count'] = len(controls[0]['frozen_heads'])
                r['edit_controls'] = cstat
                r['edit_controls_saved_summary'] = read(q / f'edit_controls_{phase}_summary.json')
                control_rows.append(dict(job=job, phase=phase, head_count=cstat['frozen_heads_count'], **cstat['paired_answer_change_minus_same_answer']))
                r['ablation'] = {}
                for origin, folder in [('iconqa', q), ('clevr', q / 'clevr_frozen_heads')]:
                    aa = list(rows(folder / f'heads_ablation_{phase}.jsonl'))
                    for x in aa:
                        x['pair_id'] = items[x['recipient_variant_id'].split('|')[0]]['counterfactual_pair_id']
                    r['ablation'][origin] = {name: {
                        'margin_drop': clustered([x for x in aa if x['head_set']==name], 'margin_drop'),
                        'still_correct': mean(x['still_correct'] for x in aa if x['head_set']==name)}
                        for name in ['top4', 'all_heads', 'random4_0']}
                r['head_means_recomputed'] = {}
                for origin, folder in [('iconqa', q), ('clevr', q / 'clevr_frozen_heads')]:
                    hh = list(rows(folder / f'heads_validate_{phase}.jsonl'))
                    summary = r['aggregate']['heads'] if origin == 'iconqa' else r['clevr_heads']
                    means = {name:clustered([x for x in hh if x['set']==name], 'normalized_recovery')['mean']
                             for name in ['all_heads'] + ['top'+k for k in summary['per_k']]}
                    assert math.isclose(means['all_heads'], summary['all_heads_mean_nr'], abs_tol=1e-12)
                    for k, v in summary['per_k'].items():
                        assert math.isclose(means['top'+k], v['top_k']['mean'], abs_tol=1e-12)
                    r['head_means_recomputed'][origin] = means
                # Recover phase-suffixed lens records directly. Legacy aggregate_run
                # only discovers lens_behavior.jsonl, so it omits these curves.
                groups = defaultdict(list)
                for x in rows(q / f'lens_{phase}.jsonl'):
                    groups[(x['scheme'], x['stage'])].append(x)
                lc = []
                for (scheme, stage), group in sorted(groups.items()):
                    d = dict(job=job, phase=phase, scheme=scheme, stage=stage, records=len(group))
                    for key in ['margin', 'mean_label_logit', 'correct_restricted_prob', 'correct_prob_full',
                                'label_mass_full', 'content_margin', 'requested_mass_8way', 'global_top_is_label', 'correct']:
                        d[key] = mean(x[key] for x in group if key in x)
                    lc.append(d)
                lens_rows.extend(lc)
                r['lens_first_positive_mean_margin'] = {s:next((x['stage'] for x in lc if x['scheme']==s and x['margin']>0),None) for s in ['letters','numbers','rare_letters']}
                r['lens_final'] = {x['scheme']:x for x in lc if x['stage'] == max(y['stage'] for y in lc)}
                counts = r['raw_counts']
                layers = 28 if job == '46804' else 36
                expected = {'patching':576*layers+256*4*(layers//2+1), 'components':384*2*layers,
                            'component_ablation':32*2*layers, 'behavior':2560,
                            'lens':1536*(layers+1), 'component_projection':1536*2*layers,
                            'edit_controls':32, 'heads_screen':64*4*(28 if job=='46804' else 32)}
                for stem,n in expected.items():
                    if stem == 'heads_screen' and phase=='confirmation':
                        continue
                    assert counts[stem+'_'+phase] == n, (job,phase,stem,counts.get(stem+'_'+phase),n)
                r['expected_counts_passed'] = expected
            elif job == '45585' or phase == 'discovery':
                for stem, count in {'patching':151040,'components':114688,'component_ablation':7168,
                                    'behavior':5120,'lens':89088,'component_projection':172032}.items():
                    assert r['raw_counts'][stem+'_'+phase] == count, (job, phase, stem)
            # Final residual, token-group and component curves with pair-based CIs.
            for stem in ['patching', 'components']:
                grouped = defaultdict(list)
                for x in rows(q / f'{stem}_{phase}.jsonl'):
                    grouped[(x['contrast_type'], x['site'], x.get('group','final'), x['layer'])].append(x)
                for (kind, site, group, layer), rr in sorted(grouped.items()):
                    if kind == 'identity':
                        stat = clustered(rr, 'recipient_margin_change')
                        metric = 'recipient_margin_change'
                    else:
                        stat = clustered(rr, 'normalized_recovery')
                        metric = 'normalized_recovery'
                    curve_rows.append(dict(job=job,phase=phase,contrast=kind,site=site,group=group,layer=layer,
                                           metric=metric,flip_to_donor=mean(x['flip_to_donor'] for x in rr if 'flip_to_donor' in x),**stat))
            print('Audited raw counts and summaries:', job, phase, flush=True)
    # B1/B4 is matched numerical reproducibility, never an independent replicate.
    p1, p4 = [BACKUPS/('job_'+j)/'results' for j in JOBS[:2]]
    matches={}
    for phase in ['discovery','confirmation']:
        for name in ['selected_contrasts.json','eligibility.json',f'behavior_{phase}.jsonl',f'lens_{phase}.jsonl',f'component_projection_{phase}.jsonl']:
            matches[phase+'/'+name] = sha(p1/phase/name)==sha(p4/phase/name)
    a,b=[read(p/'discovery/aggregate_discovery.json') for p in [p1,p4]]
    excluded={'source_git_commit','freeze_hash','frozen_from_run'}
    f1,f4=[read(p/'iconqa_freeze.json') for p in [p1,p4]]
    metrics['qwen_batch_comparison']={'byte_equal':matches,
        'same_scientific_freeze':{k:v for k,v in f1.items() if k not in excluded}=={k:v for k,v in f4.items() if k not in excluded},
        'max_curve_abs_delta':{k:max(abs(a[k][l]-b[k][l]) for l in a[k]) for k in ['attn_curve','mlp_curve','content_curve','position_curve']},
        'all_heads_abs_delta':abs(a['heads']['all_heads_mean_nr']-b['heads']['all_heads_mean_nr'])}
    metrics['small_cohort_overlap']={}
    for phase in ['discovery','confirmation']:
        a,b=[set(metrics['runs'][j]['phases'][phase]['eligibility']['pair_ids']) for j in SMALL]
        metrics['small_cohort_overlap'][phase]={'qwen':len(a),'internvl':len(b),'shared':len(a&b),
            'qwen_only':sorted(a-b),'internvl_only':sorted(b-a)}
    # Paired model difference on the same complete behavior population.
    metrics['paired_behavior_difference_qwen_minus_internvl']={}
    for phase in ['discovery','confirmation']:
        merged=[]
        a={x['variant_id']:x for x in behavior_raw['46804'][phase] if x['condition']=='clean' and x['scheme']=='letters'}
        b={x['variant_id']:x for x in behavior_raw['46912'][phase] if x['condition']=='clean' and x['scheme']=='letters'}
        assert a.keys()==b.keys()
        for k in a:
            merged.append(dict(pair_id=a[k]['pair_id'],difference=float(a[k]['correct'])-float(b[k]['correct'])))
        metrics['paired_behavior_difference_qwen_minus_internvl'][phase]=clustered(merged,'difference')
    (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    write_csv('behavior_strata.csv',behavior_rows)
    write_csv('causal_curves_pair_bootstrap.csv',curve_rows)
    write_csv('lens_curves.csv',lens_rows)
    write_csv('fixed_text_control_differences.csv',control_rows)
    plot(metrics,curve_rows,lens_rows)
    (OUT/'analysis_manifest.json').write_text(json.dumps({
        'analysis_source':str(Path(__file__).resolve()),'analysis_source_sha256':sha(Path(__file__)),
        'inputs':{j:sha(BACKUPS/('job_'+j)/'remote_inventory.json') for j in JOBS},
        'outputs':{str(f.relative_to(OUT)):sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name!='analysis_manifest.json'}},indent=2)+'\n',encoding='utf-8')
    publish_summary(metrics)
    print('Analysis complete:',OUT,flush=True)


def publish_summary(metrics):
    """Keep reviewable results in Git; full eligibility/provenance detail stays local."""
    compact = {k:v for k,v in metrics.items() if k != 'runs'}
    compact['runs'] = {}
    for job, run in metrics['runs'].items():
        entry = {k:run[k] for k in ['name','source_commit','run_state','freeze']}
        entry['seed'] = run['resume_identity']['seed']
        entry['phases'] = {}
        for phase, data in run['phases'].items():
            keep = ['behavior','raw_counts','raw_nonfinite','calibration_all_passed',
                    'checkpoints','checkpoint_artifacts_verified','replication_checks',
                    'behavior_strata','edit_controls','edit_controls_saved_summary',
                    'ablation','lens_first_positive_mean_margin','lens_final']
            d = {k:data[k] for k in keep if k in data}
            d['eligibility'] = {k:data['eligibility'][k] for k in
                                ['pairs_used','target_pairs','scanned_pairs'] if k in data['eligibility']}
            if 'aggregate' in data:
                d['aggregate'] = {k:v for k,v in data['aggregate'].items() if k != 'heads'}
            for source, key in [('iconqa_heads','aggregate'),('clevr_heads','clevr_heads')]:
                head = data.get('aggregate',{}).get('heads') if source=='iconqa_heads' else data.get(key)
                if head:
                    d[source] = {k:head[k] for k in ['candidates','per_k','all_heads_mean_nr','verdict','topk']}
            entry['phases'][phase] = d
        compact['runs'][job] = entry
    (OUT/'results_summary.json').write_text(json.dumps(compact,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    names = ['README.md','results_summary.json','behavior_strata.csv',
             'causal_curves_pair_bootstrap.csv','lens_curves.csv',
             'fixed_text_control_differences.csv','accounting_recheck_20261030.txt']
    names += [str(p.relative_to(OUT)).replace('\\','/') for p in sorted((OUT/'figures').glob('*.png'))]
    original = read(OUT/'analysis_manifest.json')
    manifest = {'analysis_script':'scripts/local/analyze_iconqa_transfer_runs.py',
                'publication_script_sha256':sha(Path(__file__)),
                'numerical_analysis_script_sha256':original['analysis_source_sha256'],
                'input_inventory_sha256':original['inputs'],
                'files':{name:sha(OUT/name) for name in names if (OUT/name).exists()},
                'local_only':['metrics.json','analysis_manifest.json','figures/*.svg'],
                'note':'Compact export of audited numerical results. Raw backups/datasets are not published.'}
    (OUT/'publication_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')


def plot(metrics, curves, lens):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
    colors={'46804':'#2463A6','46912':'#C05C20'}
    fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,j in zip(axs,SMALL):
        for kind,style in [('content','-'),('position','--')]:
            data=[r for r in curves if r['job']==j and r['phase']=='confirmation' and r['site']=='resid_post' and r['group']=='final' and r['contrast']==kind]
            ax.plot([x['layer'] for x in data],[x['mean'] for x in data],style,label=kind,color=colors[j],lw=2)
            ax.fill_between([x['layer'] for x in data],[x['ci_low'] for x in data],[x['ci_high'] for x in data],color=colors[j],alpha=.12)
        ax.axhline(.5,color='gray',lw=.8);ax.set(title=NAMES[j],xlabel='Decoder layer (zero based)',ylabel='Normalized recovery');ax.legend()
    fig.suptitle('CF-small confirmation: content and position transfer (32 source pairs/model)')
    save(fig,'01_residual_transfer')
    fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,j in zip(axs,SMALL):
        a=metrics['runs'][j]['phases']['confirmation'];h=a['aggregate']['heads'];c=a['clevr_heads']
        for source,summary,style in [('IconQA-selected',h,'-o'),('CLEVR-frozen',c,'--s')]:
            kk=sorted(map(int,summary['per_k']))
            ax.plot(kk,[summary['per_k'][str(k)]['top_k']['mean'] for k in kk],style,label=source,color=colors[j],alpha=1 if source=='IconQA-selected' else .6)
        ax.set(title=NAMES[j],xlabel='Number of patched heads',ylabel='Normalized recovery',xticks=[1,2,4,8,16]);ax.legend()
    fig.suptitle('CF-small confirmation: head transfer; all-head reference differs by frozen layer set')
    save(fig,'02_head_transfer')
    fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for j in SMALL:
        data=[r for r in curves if r['job']==j and r['phase']=='confirmation' and r['site']=='resid_post' and r['group']=='image' and r['contrast']=='content']
        axs[0].plot([r['layer'] for r in data],[r['mean'] for r in data],label=NAMES[j],color=colors[j]);axs[0].fill_between([r['layer'] for r in data],[r['ci_low'] for r in data],[r['ci_high'] for r in data],alpha=.12,color=colors[j])
        ls=[r for r in lens if r['job']==j and r['phase']=='confirmation' and r['scheme']=='letters']
        axs[1].plot([r['stage'] for r in ls],[r['margin'] for r in ls],label=NAMES[j],color=colors[j])
    axs[0].set(title='Causal image-token patches',xlabel='Decoder layer (zero based)',ylabel='Normalized recovery');axs[0].legend()
    axs[1].axhline(0,color='gray',lw=.8);axs[1].set(title='Observational A/B/C/D lens',xlabel='Residual stage (0 = input)',ylabel='Correct minus best incorrect logit');axs[1].legend()
    save(fig,'03_image_and_lens')


def save(fig,name):
    for suffix in ['png','svg']:
        fig.savefig(OUT/'figures'/f'{name}.{suffix}',bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    main()
