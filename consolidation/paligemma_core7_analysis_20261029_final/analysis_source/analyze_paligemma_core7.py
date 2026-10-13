"""Analyze immutable PKGPU Pali evidence; no torch/model imports or inference.

Writes a new analysis package. Raw archive/extracted records are never edited.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, platform, shutil, subprocess
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
EXPORT = REPO/'results/pkgpu_pali10b_analysis_20261029'
INPUT = EXPORT/'extracted'
RESULT = INPUT/'results/pali10b_memoryfix_20261026'
RUN = RESULT/'run_20261026T114918_068380Z'
SEED = 20261027  # same additional pair estimand as the three-model synthesis
MODEL = 'paligemma2_10b_mix'

def dump(p, x):
    p.write_text(json.dumps(x, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')

def csvout(p, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):
    with p.open(encoding='utf-8-sig') as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def table(p):
    with p.open(newline='', encoding='utf-8-sig') as f: return list(csv.DictReader(f))

def cluster(groups):
    v=np.array([mean(groups[k]) for k in sorted(groups)], dtype=float)
    rng=np.random.default_rng(SEED)
    draws=v[rng.integers(0,len(v),(2000,len(v)))].mean(axis=1)
    lo,hi=np.quantile(draws,[.025,.975])
    return dict(estimate=float(v.mean()),ci_low=float(lo),ci_high=float(hi),n_independent=len(v),n_rows=sum(map(len,groups.values())))

def first(curve):
    return next((int(r['layer']) for r in sorted(curve,key=lambda r:int(r['layer'])) if float(r.get('estimate',r.get('mean',0)))>=.5),None)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output', default='consolidation/paligemma_core7_analysis_20261029');args=ap.parse_args()
    out=REPO/args.output;out.mkdir(parents=True,exist_ok=False)
    for n in ['tables','figures','evidence']: (out/n).mkdir()
    checks=[]
    def check(name, passed, detail=''):
        checks.append(dict(check=name,passed=bool(passed),detail=detail))
        if not passed: raise AssertionError(f'{name}: {detail}')
    def close(name,a,b):check(name,math.isclose(float(a),float(b),rel_tol=1e-8,abs_tol=1e-9),f'{a} vs {b}')
    # One evidence pass verifies exported content, rather than rerunning models/tests.
    archive=read(INPUT/'ARCHIVE_MANIFEST.json')
    for record in archive['files']:
        p=INPUT/record['path'];check('archive member hash '+record['path'],p.is_file() and sha(p)==record['sha256'])
    freeze=read(RUN/'freeze.json');confirm=read(RUN/'confirmation/confirmation_report.json')
    identity=dict(freeze);identity.pop('freeze_hash')
    check('freeze self hash',hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]==freeze['freeze_hash'])
    check('confirmation freeze binding',confirm['freeze_hash']==freeze['freeze_hash'])
    comparison=read(RESULT/'compare_20261026T114219_608942Z/comparison.json')
    check('original optimized comparison passed',comparison['all_passed'] is True)
    dump(out/'evidence/comparison.json',comparison)
    summaries={}; inventories=[]; coverage=[]; pair_rows=[]; headsets=[]; headabl=[]; headindividual=[]; tidy=[]; figures=[]; sampleids={}; ablations=[]
    def metric(method,phase,metric,estimate,source,**extra):
        tidy.append(dict(model=MODEL,method=method,phase=phase,metric=metric,estimate=estimate,source=source.relative_to(REPO).as_posix(),availability='available',**extra))
    def raw_audit(p,expected=None,identity_fields=None):
        count=0;keys=set();pairs=set();variants=set()
        for r in rows(p):
            count+=1
            for k,v in r.items():
                if isinstance(v,float):check('finite '+p.name+'/'+k,math.isfinite(v)) if not math.isfinite(v) else None
            if identity_fields:
                key=tuple(r.get(k) for k in identity_fields)
                if key in keys:check('unique raw identity '+p.name,False,str(key))
                keys.add(key)
            if r.get('pair_id'):pairs.add(r['pair_id'])
            if r.get('recipient_variant_id'):variants.add(r['recipient_variant_id'])
        if expected is not None:check('raw count '+p.name,count==expected,f'{count} vs {expected}')
        inventories.append(dict(phase=p.parent.name,file=p.name,rows=count,expected_rows=expected,distinct_pair_ids=len(pairs),distinct_recipient_variants=len(variants),sha256=sha(p)))
        return count
    # Preserve all original tables; export source references in a mergeable long table.
    fields=('contrast_id','method','site','layer','group','set','head','head_set','recipient_variant_id','donor_variant_id')
    for phase in ['discovery','confirmation']:
        root=RUN/phase;run=read(root/'run.json');agg=read(root/f'aggregate_{phase}.json');heads=read(root/f'heads_{phase}_summary.json');patch=read(root/f'patching_{phase}_summary.json');comp=read(root/f'components_{phase}_summary.json')
        check('phase complete '+phase,run['status']=='complete')
        cal=read(root/'calibration.json');check('calibration '+phase,cal['all_passed'] is True)
        for gate,g in cal['gates'].items():
            if isinstance(g,dict) and 'passed' in g:check('calibration gate '+phase+'/'+gate,g['passed'])
        summaries[phase]={'aggregate':{k:v for k,v in agg.items() if k!='heads'},'heads':{k:v for k,v in heads.items() if k not in ['cases','individual']},'calibration':cal}
        figures.append(dict(phase=phase,listed=len(run['figures']),direct_paths_found=sum((root/f).is_file() for f in run['figures']),figure_subfolder_found=sum((root/'figures'/f).is_file() and (root/'figures'/f).stat().st_size>0 for f in run['figures'])))
        check('all figures present under figures/ '+phase,figures[-1]['listed']==figures[-1]['figure_subfolder_found'])
        for method,data in [('M1/M6',patch),('M2',comp),('M3',heads)]:
            c=data['cases'];sampleids[phase+'/'+method]=c
            for kind,n in c['cases_by_type'].items():coverage.append(dict(phase=phase,method=method,contrast_type=kind,directed_cases=n,pairs=c['selected_pairs_by_type'][kind],eligible_pairs=c['eligible_pairs'],selection=c['limited_case_selection']))
        ncases=sum(heads['cases']['cases_by_type'].values());nsets=len(heads['topk'])*(1+run['profile']['random_sets'])+1
        expected={f'patching_{phase}.jsonl':patch['m1_rows']+patch['group_patching']['rows'],f'components_{phase}.jsonl':comp['patch_rows'],f'component_ablation_{phase}.jsonl':comp['ablation_rows'],f'heads_validate_{phase}.jsonl':ncases*(len(heads['candidates'])+nsets),f'heads_ablation_{phase}.jsonl':heads['ablation_rows']}
        if phase=='discovery':
            expected.update(run['expected_observational_rows']);expected[f'heads_screen_{phase}.jsonl']=ncases*len(heads['head_layers'])*run['num_heads']
        for filename,n in expected.items():raw_audit(root/filename,n,fields if filename.startswith(('patching_','components_','heads_validate_','heads_screen_','component_ablation_','heads_ablation_')) else None)
        for p in sorted((root/'tables').glob('*.csv')):
            tr=table(p);shutil.copyfile(p,out/'tables'/p.name)
            for r in tr:
                method='M1' if p.name.startswith('m1_') else 'M2' if p.name.startswith('m2_') else 'M6' if p.name.startswith('m6_') else 'M4/M5/M7'
                if 'mean' in r:
                    metric(method,'shared_behavior' if p.name in ['m4_m5_m7_lens.csv','m2_projection.csv'] else phase,r.get('metric',''),float(r['mean']),p,**{k:(float(v) if k in ['ci_low','ci_high'] else v) for k,v in r.items() if k not in ['mean','metric']},interval_method='original directed-row bootstrap (2000, Python random; not pair-clustered)',evidence_level='original aggregate table')
                else:
                    for k,v in r.items():
                        if k.startswith('frac_'):metric(method,phase,k,float(v),p,**{a:b for a,b in r.items() if not a.startswith('frac_')},interval_method='none supplied',evidence_level='original aggregate table')
        # Cluster by semantic pair for causal curves; preserve original directed means separately.
        for name in ['patching','components']:
            groups=defaultdict(lambda:defaultdict(list));counts=Counter()
            for r in rows(root/f'{name}_{phase}.jsonl'):
                value=r.get('normalized_recovery')
                if value is None or not math.isfinite(value):continue
                key=(r['contrast_type'],r.get('site',''),r.get('group','final'),r['layer'])
                groups[key][r['pair_id']].append(value);counts[key]+=1
            for (kind,site,group,layer),g in sorted(groups.items()):
                v=[x for xs in g.values() for x in xs]
                rec=dict(model=MODEL,method='M2' if name=='components' else 'M1' if group=='final' else 'M6',phase=phase,contrast_type=kind,site=site,group=group,layer=layer,directed_mean=mean(v),**cluster(g),interval_method='equal-pair PCG64 percentile bootstrap; 2000 draws; seed 20261027',source=(root/f'{name}_{phase}.jsonl').relative_to(REPO).as_posix())
                # Random-pair donor dependencies not covered by recipient-pair clustering.
                rec['dependency_caveat']='recipient-pair only; unrelated donor reuse not modeled' if kind=='random_pair' else ''
                pair_rows.append(rec)
        # Recompute all M3 set and individual means directly, including original CI algorithm.
        hg=defaultdict(lambda:defaultdict(list));hrows=defaultdict(list)
        for r in rows(root/f'heads_validate_{phase}.jsonl'):hg[r['set']][r['pair_id']].append(r['normalized_recovery']);hrows[r['set']].append(r)
        hmean={k:mean([mean(g[p]) for p in sorted(g)]) for k,g in hg.items()}
        close('M3 all-head mean '+phase,hmean['all_heads'],heads['all_heads_mean_nr'])
        for k,v in heads['per_k'].items():
            close('M3 selected mean '+phase+'/'+k,hmean['top'+k],v['top_k']['mean'])
            randoms=sorted(hmean[f'random{k}_{j}'] for j in range(20));close('M3 random p95 '+phase+'/'+k,randoms[18],v['random_p95'])
            close('M3 fraction '+phase+'/'+k,hmean['top'+k]/hmean['all_heads'],v['fraction_of_all_heads'])
            rec=dict(phase=phase,k=int(k),**v['top_k'],random_mean=v['random_mean'],random_p95=v['random_p95'],all_selected_layer_heads=hmean['all_heads'],fraction_of_all_heads=v['fraction_of_all_heads'],beats_random_p95=v['beats_random_p95'],layers=str(heads.get('head_layers',freeze['head_layers'])),network_heads=42*16,denominator_heads=4*16)
            headsets.append(rec)
            for metricname,value in [('selected_head_recovery',v['top_k']['mean']),('matched_random_p95',v['random_p95']),('fraction_selected_layer_all_heads',v['fraction_of_all_heads'])]:metric('M3',phase,metricname,value,root/f'heads_{phase}_summary.json',k=int(k),n_rows=ncases,n_independent=v['top_k']['n_clusters'],independent_unit='semantic pair',ci_low=v['top_k']['ci_low'] if metricname=='selected_head_recovery' else '',ci_high=v['top_k']['ci_high'] if metricname=='selected_head_recovery' else '',interval_method='original pair bootstrap' if metricname=='selected_head_recovery' else 'none; random p95 is an empirical reference')
        for k,v in heads['individual'].items():
            close('individual head '+phase+'/'+k,hmean[k],v['mean']);headindividual.append(dict(phase=phase,head=k,**v))
        abl=defaultdict(list)
        for r in rows(root/f'heads_ablation_{phase}.jsonl'):abl[r['head_set']].append(r)
        for k,rs in abl.items():headabl.append(dict(phase=phase,head_set=k,n=len(rs),margin_drop=mean(x['margin_drop'] for x in rs),still_correct=mean(x['still_correct'] for x in rs),note='recipient-weighted descriptive; donor reuse not clustered'))
        for k in ['position_transfer_layer','content_transfer_layer','attn_peak_layer','mlp_peak_layer','attn_peak_nr','mlp_peak_nr','letters_readable_stage']:
            if k in agg:metric('M4' if k=='letters_readable_stage' else 'M2' if k.startswith(('attn','mlp')) else 'M1', 'shared_behavior' if k=='letters_readable_stage' else phase,k,agg[k],root/f'aggregate_{phase}.json',units='residual stage' if k=='letters_readable_stage' else 'zero-based layer' if k.endswith('layer') else 'normalized recovery')
    # Disjoint discovery and confirmation for each actual selected method cohort.
    overlap=[]
    for method in ['M1/M6','M2','M3']:
        a=set(sampleids['discovery/'+method]['pair_ids']);b=set(sampleids['confirmation/'+method]['pair_ids'])
        check('phase disjoint '+method,not a&b);overlap.append(dict(method=method,discovery=len(a),confirmation=len(b),overlap=len(a&b)))
    # Raw table means and counts independently checked against generated summaries.
    for phase in ['discovery','confirmation']:
        for basename,method,group in [('m1_residual','M1','final'),('m2_components','M2','final'),('m6_token_groups','M6',None)]:
            for r in table(RUN/phase/'tables'/f'{basename}_{phase}.csv'):
                if r['metric']!='normalized_recovery':continue
                candidates=[p for p in pair_rows if p['phase']==phase and p['method']==method and p['contrast_type']==r['contrast_type'] and p['layer']==int(r['layer']) and (not r.get('site') or p['site']==r['site']) and (not r.get('group') or p['group']==r['group'])]
                check('raw/table curve key '+phase+'/'+basename,len(candidates)==1)
                close('raw/table mean '+phase+'/'+basename,candidates[0]['directed_mean'],r['mean']);check('raw/table n',candidates[0]['n_rows']==int(r['n']))
    # Behavior: item-clustered estimates, paired image controls, clean robustness and error profiles.
    behavior=[r for r in rows(RUN/'discovery/behavior_behavior.jsonl') if r['split']=='behavior']
    bs=read(RUN/'discovery/behavior_behavior_summary.json');behavior_rows=[];profiles=[];be_groups={}
    def bstat(label,rs,field='correct'):
        g=defaultdict(list)
        for r in rs:g[r['item_id']].append(float(r[field]))
        stats=cluster(g);behavior_rows.append(dict(metric=label,**stats,unit='item',field=field));be_groups[label]=g
        return stats
    clean=[r for r in behavior if r['condition']=='clean'];letters=[r for r in clean if r['scheme']=='letters']
    for cond in ['clean','blank','shuffled']:
        rs=[r for r in behavior if r['condition']==cond and r['scheme']=='letters'];st=bstat(cond+'_letters_accuracy',rs)
        close('behavior '+cond,st['estimate'],bs['letters_accuracy' if cond=='clean' else cond+'_accuracy'])
    for cond in ['blank','shuffled']:
        g={k:[mean(be_groups['clean_letters_accuracy'][k])-mean(v)] for k,v in be_groups[cond+'_letters_accuracy'].items()};behavior_rows.append(dict(metric='clean_minus_'+cond,**cluster(g),unit='paired item',field='correct difference'))
    robustids={r['item_id'] for r in clean if r['scheme']!='letters'}
    for scheme in ['letters','numbers','rare_letters']:
        rs=[r for r in clean if r['scheme']==scheme];rob=[r for r in rs if r['item_id'] in robustids]
        for label,data,field in [('accuracy',rob,'correct'),('label_mass_full',rs,'label_mass_full'),('global_argmax_valid',rs,'global_top_is_label')]:bstat(scheme+'_'+label,data,field)
    for dim in ['family','depth','position']:
        for v in sorted({str(r[dim]) for r in letters}):
            rs=[r for r in letters if str(r[dim])==v];g=defaultdict(list)
            for r in rs:g[r['item_id']].append(float(r['correct']))
            profiles.append(dict(dimension=dim,category=v,**cluster(g)))
    comps=list(rows(RUN/'discovery/compliance_behavior.jsonl'))
    summaries['behavior']=bs;summaries['behavior_counts']=dict(total_rows=len(behavior),clean_rows=len(clean),letters_rows=len(letters),robust_items=len(robustids),compliance=len(comps),scheme_counts={s:dict(rows=sum(r['scheme']==s for r in clean),items=len({r['item_id'] for r in clean if r['scheme']==s})) for s in ['letters','numbers','rare_letters']})
    # Additional descriptive lens quantities omitted from original CSV; item-clustered.
    lg=defaultdict(lambda:defaultdict(list))
    for r in rows(RUN/'discovery/lens_behavior.jsonl'):
        for field in ['margin','mean_label_logit','correct_restricted_prob','content_margin','label_mass_full','correct_prob_full','global_top_is_label','correct','requested_mass_8way','requested_mass_full','default_mass_full']:
            if field in r:lg[(r['scheme'],r['stage'],field)][r['item_id']].append(float(r[field]))
    lens=[dict(scheme=s,stage=st,metric=m,**cluster(g)) for (s,st,m),g in sorted(lg.items())]
    # Runtime from immutable checkpoint timestamps in archive, not extraction mtimes.
    import zipfile,datetime
    times=[]
    with zipfile.ZipFile(EXPORT/'pali10b_core7_analysis_20261029_complete.zip') as z:
        for item in z.infolist():
            if '/run_20261026T114918_068380Z/' in item.filename and '/checkpoints/' in item.filename:
                t=datetime.datetime(*item.date_time);times.append(dict(block='/'.join(item.filename.split('/')[-3:]),server_local_time_IST=t.isoformat(),seconds_since_previous=None))
    times.sort(key=lambda r:r['server_local_time_IST'])
    for i in range(1,len(times)):times[i]['seconds_since_previous']=(datetime.datetime.fromisoformat(times[i]['server_local_time_IST'])-datetime.datetime.fromisoformat(times[i-1]['server_local_time_IST'])).total_seconds()
    sensitivity=[]
    for phase in ['discovery','confirmation']:
        for kind in ['position','content']:
            rr=[r for r in pair_rows if r['phase']==phase and r['method']=='M1' and r['contrast_type']==kind]
            sensitivity.append(dict(phase=phase,contrast_type=kind,directed_first_crossing=summaries[phase]['aggregate'][kind+'_transfer_layer'],equal_pair_first_crossing=first(rr),pair_counts=sorted({r['n_independent'] for r in rr})))
    for name,data in [('metrics_tidy',tidy),('sample_coverage',coverage),('raw_inventory',inventories),('pair_reanalysis',pair_rows),('head_sets',headsets),('head_individual',headindividual),('head_ablation',headabl),('behavior_clustered',behavior_rows),('behavior_profiles',profiles),('lens_item_clustered',lens),('checkpoint_timeline',times),('weighting_sensitivity',sensitivity),('figure_path_audit',figures),('phase_overlap',overlap)]:csvout(out/'tables'/f'{name}.csv',data)
    dump(out/'evidence/sample_inventory.json',sampleids)
    dump(out/'evidence/freeze.json',freeze);dump(out/'evidence/confirmation_report.json',confirm)
    summaries['weighting_sensitivity']=sensitivity;summaries['figure_paths']=figures;summaries['comparison']={k:v for k,v in comparison.items() if k not in ['checks','input_identity','reference_selection','optimized_selection','variant_ids','contrast_ids']}
    dump(out/'summary.json',summaries)
    make_figures(out,pair_rows,headsets,behavior_rows,lens)
    dump(out/'validation.json',dict(scope='saved-evidence content/count/mean/freeze and figure-path verification; no model tests',passed=all(c['passed'] for c in checks),checks=checks))
    sources=[p for p in INPUT.rglob('*') if p.is_file()]
    dump(out/'manifest.json',dict(analysis='PALI-CORE7-ANALYSIS-20261029',model=MODEL,seed=SEED,bootstrap_samples=2000,archive_sha256=sha(EXPORT/'pali10b_core7_analysis_20261029_complete.zip'),analysis_script_sha256=sha(Path(__file__)),python=platform.python_version(),numpy=np.__version__,inputs=[dict(path=p.relative_to(REPO).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in sources],no_model_execution=True))
    enrich(out)
    print(json.dumps(dict(output=str(out),checks=len(checks),raw_rows=sum(r['rows'] for r in inventories),metrics=len(tidy),pair_endpoints=len(pair_rows),figures=len(list((out/'figures').glob('*.png'))))))

def make_figures(out,pr,hs,bs,lens):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors={'discovery':'#2166ac','confirmation':'#b35806'}
    def save(fig,name):
        fig.tight_layout();fig.savefig(out/'figures'/f'{name}.png',dpi=170);fig.savefig(out/'figures'/f'{name}.svg');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for ax,kind in zip(axs,['position','content']):
        for phase in colors:
            r=sorted([p for p in pr if p['phase']==phase and p['method']=='M1' and p['contrast_type']==kind],key=lambda p:p['layer'])
            ax.plot([x['layer'] for x in r],[x['estimate'] for x in r],color=colors[phase],label=phase+' equal pair')
            ax.fill_between([x['layer'] for x in r],[x['ci_low'] for x in r],[x['ci_high'] for x in r],color=colors[phase],alpha=.14)
            ax.plot([x['layer'] for x in r],[x['directed_mean'] for x in r],color=colors[phase],ls=':',alpha=.7)
        ax.axhline(.5,color='gray',ls='--');ax.set(title=kind+' contrasts',xlabel='Decoder layer (zero based)',ylabel='Normalized recovery');ax.legend(fontsize=8)
    fig.suptitle('M1 residual transfer: equal-pair 95% intervals; dotted = original directed mean');save(fig,'01_residual_transfer')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for ax,phase in zip(axs,colors):
        for site,col in [('attn','#2166ac'),('mlp','#b35806')]:
            r=sorted([p for p in pr if p['phase']==phase and p['method']=='M2' and p['contrast_type']=='position' and p['site']==site],key=lambda p:p['layer'])
            ax.plot([x['layer'] for x in r],[x['estimate'] for x in r],color=col,label=site)
            ax.fill_between([x['layer'] for x in r],[x['ci_low'] for x in r],[x['ci_high'] for x in r],color=col,alpha=.15)
        ax.set(title=phase,xlabel='Decoder layer',ylabel='Position normalized recovery');ax.legend()
    fig.suptitle('M2 component transfer: equal-pair means and 95% intervals');save(fig,'02_component_transfer')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for phase in colors:
        r=sorted([h for h in hs if h['phase']==phase],key=lambda h:h['k']);x=np.arange(len(r))
        axs[0].errorbar(x,[h['mean'] for h in r],yerr=[[h['mean']-h['ci_low'] for h in r],[h['ci_high']-h['mean'] for h in r]],label=phase,color=colors[phase],marker='o')
        axs[0].plot(x,[h['random_p95'] for h in r],ls=':',color=colors[phase])
        axs[1].plot(x,[h['fraction_of_all_heads'] for h in r],label=phase,color=colors[phase],marker='o')
    for ax in axs:ax.set_xticks(range(5),[1,2,4,8,16]);ax.set_xlabel('Selected head set size');ax.legend()
    axs[0].set_ylabel('Normalized recovery (pair mean; original 95% CI)');axs[1].set_ylabel('Effect / all 64 heads in four selected layers');axs[1].axhline(.5,color='gray',ls='--')
    fig.suptitle('M3 frozen heads: dotted = empirical matched-random p95, not a CI');save(fig,'03_head_sets')
    fig,axs=plt.subplots(1,3,figsize=(14,4.3))
    for ax,m in zip(axs,['mean_label_logit','margin','label_mass_full']):
        for scheme in ['letters','numbers','rare_letters']:
            r=sorted([x for x in lens if x['metric']==m and x['scheme']==scheme],key=lambda x:x['stage']);ax.plot([x['stage'] for x in r],[x['estimate'] for x in r],label=scheme)
        ax.set(title=m.replace('_',' '),xlabel='Residual stage (0 = decoder input)');ax.legend(fontsize=8)
    fig.suptitle('M4/M7 shared observational readouts: untuned final norm and vocabulary projection');save(fig,'04_lens_readouts')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for ax,phase in zip(axs,colors):
        for group in ['image','question','options','labels']:
            r=sorted([p for p in pr if p['phase']==phase and p['method']=='M6' and p['contrast_type']=='content' and p['group']==group],key=lambda p:p['layer'])
            ax.plot([x['layer'] for x in r],[x['estimate'] for x in r],label=group)
        ax.set(title=phase,xlabel='Decoder layer',ylabel='Content recovery (equal-pair mean)');ax.legend()
    fig.suptitle('M6 token-group patches: groups differ in size; curves are not route edges');save(fig,'05_token_groups')
    fig,ax=plt.subplots(figsize=(8,4.7));r=[x for x in bs if x['metric'] in ['clean_letters_accuracy','blank_letters_accuracy','shuffled_letters_accuracy']]
    ax.bar(range(3),[x['estimate'] for x in r],yerr=[[x['estimate']-x['ci_low'] for x in r],[x['ci_high']-x['estimate'] for x in r]],color=['#2166ac','#aaaaaa','#b35806'],capsize=5)
    ax.set_xticks(range(3),['Clean','Blank','Shuffled']);ax.axhline(.25,color='gray',ls='--');ax.set(ylabel='Restricted four-label accuracy',ylim=(0,.7),title='Image dependence: 256 items, four letter layouts each\n95% item-cluster bootstrap intervals');save(fig,'06_behavior_controls')

def enrich(out):
    """Additional interpretation diagnostics and original-table consistency.

    May also be run on the existing derived package without repeating the main
    evidence pass. This does not touch the archived campaign.
    """
    checks=[];denoms=[];tails=[];identity=[];metric_rows=[];ablation_pairs=[]
    for phase in ['discovery','confirmation']:
        root=RUN/phase
        for rawname,prefixes in [(f'patching_{phase}.jsonl',['m1_residual','m6_token_groups','m6_outcomes']),(f'components_{phase}.jsonl',['m2_components']),(f'component_ablation_{phase}.jsonl',['m2_ablation'])]+([('lens_behavior.jsonl',['m4_m5_m7_lens']),('component_projection_behavior.jsonl',['m2_projection'])] if phase=='discovery' else []):
            specs=[]
            for pref in prefixes:
                path=root/'tables'/(pref+'.csv' if pref in ['m4_m5_m7_lens','m2_projection'] else pref+'_'+phase+'.csv')
                trs=table(path);keys=[k for k in trs[0] if k not in ['mean','n','ci_low','ci_high','metric'] and not k.startswith('frac_')]
                specs.append((pref,trs,keys,defaultdict(list),defaultdict(Counter)))
            case_seen=set();dg=defaultdict(list);rv=defaultdict(list);ir=defaultdict(list)
            for r in rows(root/rawname):
                if rawname.startswith(('patching_','components_')):
                    if r.get('normalized_recovery') is not None:
                        key=(r['contrast_type'],r.get('site',''),r.get('group','final'),r['layer']);rv[key].append(r['normalized_recovery'])
                        if r['contrast_id'] not in case_seen:
                            case_seen.add(r['contrast_id']);dg[r['contrast_type']].append(r['donor_diff']-r['recipient_diff'])
                    if r.get('contrast_type')=='identity':ir[r['layer']].append(r['recipient_margin_change'])
                for pref,trs,keys,vals,outcomes in specs:
                    if pref=='m1_residual' and r.get('group')!='final':continue
                    if pref=='m6_token_groups' and r.get('group')=='final':continue
                    ident=tuple(str(r.get(k,'')) for k in keys)
                    if pref=='m6_outcomes':
                        if r.get('contrast_type')=='content_and_position' and r.get('group')=='final' and 'outcome' in r:outcomes[ident][r['outcome']]+=1
                    else:
                        for metric in {tr['metric'] for tr in trs}:
                            if metric in r:vals[(ident,metric)].append(float(r[metric]))
            for pref,trs,keys,vals,outcomes in specs:
                for tr in trs:
                    ident=tuple(tr[k] for k in keys)
                    if pref=='m6_outcomes':
                        counts=outcomes[ident];n=sum(counts.values());ok=n==int(tr['n']) and all(math.isclose(counts[k]/n,float(tr['frac_'+k]),abs_tol=1e-9) for k in ['position','content','none','other'])
                    else:
                        v=vals[(ident,tr['metric'])];ok=len(v)==int(tr['n']) and math.isclose(mean(v),float(tr['mean']),rel_tol=1e-8,abs_tol=1e-9)
                    checks.append(dict(table=pref,phase=phase,key=str(ident),metric=tr.get('metric','outcomes'),passed=ok))
                    if not ok:raise AssertionError(f'raw aggregate mismatch {phase}/{pref}/{ident}')
            for kind,v in dg.items():denoms.append(dict(phase=phase,raw=rawname,contrast_type=kind,n_cases=len(v),minimum=min(v),maximum=max(v),negative=sum(x<0 for x in v),abs_below_0_1=sum(abs(x)<.1 for x in v),abs_below_0_5=sum(abs(x)<.5 for x in v),abs_below_1=sum(abs(x)<1 for x in v),median=float(np.median(v))))
            for (kind,site,group,layer),v in rv.items():tails.append(dict(phase=phase,raw=rawname,contrast_type=kind,site=site,group=group,layer=layer,n=len(v),median=float(np.median(v)),minimum=min(v),maximum=max(v),q05=float(np.quantile(v,.05)),q95=float(np.quantile(v,.95)),below_zero=sum(x<0 for x in v),above_one=sum(x>1 for x in v),abs_above_5=sum(abs(x)>5 for x in v)))
            for l,v in ir.items():identity.append(dict(phase=phase,layer=l,n=len(v),mean_margin_change=mean(v),max_abs_margin_change=max(map(abs,v))))
    variants={r['variant_id']:r for r in rows(INPUT/'data/clevr_mcq4_v3_official3k_cuda45pairs/variants.jsonl')}
    # Ablation pair IDs can be recovered from the input variants, not fabricated.
    for phase in ['discovery','confirmation']:
        g=defaultdict(lambda:defaultdict(list));counts=defaultdict(set)
        for raw in [f'component_ablation_{phase}.jsonl',f'heads_ablation_{phase}.jsonl']:
            for r in rows(RUN/phase/raw):
                var=variants[r['recipient_variant_id']];pair=var['counterfactual_pair_id'];key=(raw,r.get('site'),r.get('layer',''),r.get('head_set',''))
                g[key][pair].append(float(r['margin_drop']));counts[key].add(r['recipient_variant_id'])
        for (raw,site,layer,headset),groups in g.items():ablation_pairs.append(dict(phase=phase,raw=raw,site=site,layer=layer,head_set=headset,**cluster(groups),n_recipients=len(counts[(raw,site,layer,headset)]),caveat='recipient-pair clustered sensitivity; unrelated donor reuse not modeled; first 128 sorted correct recipients'))
    for name,data in [('normalization_denominators',denoms),('recovery_distribution',tails),('identity_controls',identity),('ablation_recipient_pair_sensitivity',ablation_pairs)]:csvout(out/'tables'/f'{name}.csv',data)
    dump(out/'supplemental_validation.json',dict(passed=True,checks=checks,scope='all original CSV counts/means/outcome fractions re-derived from raw; no model execution'))
    # Compact cross-model joining surface, preserving original estimands.
    bs=read(RUN/'discovery/behavior_behavior_summary.json');freeze=read(RUN/'freeze.json')
    for key in ['letters_accuracy','blank_accuracy','shuffled_accuracy','clean_minus_shuffled','compliance_rate','generated_accuracy','strict_item_consistency']:
        metric_rows.append(dict(model=MODEL,method='behavior',phase='shared_behavior',metric=key,estimate=bs[key],n_items=256,n_rows=256 if key in ['generated_accuracy','compliance_rate','strict_item_consistency'] else 1024,weighting='original layout mean except named item/generation outcomes',source='discovery/behavior_behavior_summary.json'))
    for phase in ['discovery','confirmation']:
        agg=read(RUN/phase/f'aggregate_{phase}.json');hs=read(RUN/phase/f'heads_{phase}_summary.json')
        for key in ['position_transfer_layer','content_transfer_layer','attn_peak_layer','mlp_peak_layer','attn_peak_nr','mlp_peak_nr']:
            metric_rows.append(dict(model=MODEL,method='M2' if key.startswith(('attn','mlp')) else 'M1',phase=phase,metric=key,estimate=agg[key],n_independent=76 if phase=='discovery' else 61,weighting='original directed contrast',source=f'{phase}/aggregate_{phase}.json'))
        for k,v in hs['per_k'].items():
            metric_rows.append(dict(model=MODEL,method='M3',phase=phase,metric='selected_head_recovery',k=k,estimate=v['top_k']['mean'],ci_low=v['top_k']['ci_low'],ci_high=v['top_k']['ci_high'],n_rows=96,n_independent=v['top_k']['n_clusters'],weighting='equal semantic pair',random_p95=v['random_p95'],fraction_selected_layer_all_heads=v['fraction_of_all_heads'],denominator='64 heads in four selected layers; not 672 network heads',source=f'{phase}/heads_{phase}_summary.json'))
    csvout(out/'tables/cross_model_join.csv',metric_rows)
    csvout(out/'tables/method_coverage.csv',[dict(model=MODEL,method=m,measurement=desc,discovery='available',confirmation=conf,note=note) for m,desc,conf,note in [
        ('M1','residual causal patching','held-out causal available','76/61 selected pairs; original position-first order does not replicate'),
        ('M2','component patching/resample ablation plus descriptive projection','held-out causal available; projection shared','attention dominance replicates; compare estimands separately'),
        ('M3','frozen head sets and matched-random controls','held-out causal available','96 directions / 96 discovery pairs and 95 confirmation pairs; sparse replicates'),
        ('M4','untuned logit lens','shared observational only','256 items; three schemes; not independently repeated in confirmation'),
        ('M5','content first-token lens and overlapping content/position causal factorial','lens shared; held-out factorial available','not a semantic full-answer decoder; no positive clustered mean content margin stage'),
        ('M6','token-group patching and factorial outcomes','held-out causal available','group sizes differ; normalized semantic-content effects have extreme tails'),
        ('M7','alphabet robustness and requested/default competition','shared observational only','letters/numbers/rare letters only; no punctuation result')]])
    # Add complete factorial and requested/default curves without hiding unfavorable scales.
    fig,axs=plt.subplots(1,2,figsize=(12,4.6))
    for ax,phase in zip(axs,['discovery','confirmation']):
        rr=table(out/'tables'/f'm6_outcomes_{phase}.csv')
        for k in ['position','content','none','other']:ax.plot([int(r['layer']) for r in rr],[float(r['frac_'+k]) for r in rr],label=k)
        ax.set(title=phase,xlabel='Decoder layer',ylabel='Fraction of directed joint contrasts',ylim=(-.02,1.02));ax.legend()
    fig.suptitle('M5/M6 joint content-and-position outcomes (original directed weighting)');fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(out/'figures'/f'07_factorial_outcomes.{ext}',dpi=170)
    plt.close(fig)
    ll=table(out/'tables/lens_item_clustered.csv');fig,axs=plt.subplots(1,2,figsize=(12,4.6))
    for ax,metric in zip(axs,['requested_mass_8way','content_margin']):
        for scheme in (['numbers','rare_letters'] if metric=='requested_mass_8way' else ['letters','numbers','rare_letters']):
            rr=sorted([r for r in ll if r['metric']==metric and r['scheme']==scheme],key=lambda r:int(r['stage']))
            ax.plot([int(r['stage']) for r in rr],[float(r['estimate']) for r in rr],label=scheme)
        ax.set(title='Requested alphabet / union of eight labels' if metric=='requested_mass_8way' else 'Correct semantic first-token margin',xlabel='Residual stage');ax.legend()
        if metric=='content_margin':ax.axhline(0,color='gray',ls='--')
    fig.suptitle('M7 alphabet competition and M5 content readout are distinct diagnostics');fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(out/'figures'/f'08_alphabets_content.{ext}',dpi=170)
    plt.close(fig)
    manifest=read(out/'manifest.json');manifest['analysis_script_sha256']=sha(Path(__file__));dump(out/'manifest.json',manifest)
    print(json.dumps(dict(supplemental_raw_table_checks=len(checks),passed=True,figures=8)))

if __name__=='__main__':main()
