"""Combine audited CF-small Qwen, InternVL and LLaVA evidence without inference.

The prior Qwen/InternVL and LLaVA audits remain immutable. This synthesis reads
their preserved raw records for harmonized source-pair estimates and records all
input hashes. LLaVA defaults to the sibling Jarvis worktree; override --llava-root.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'consolidation/iconqa_three_model_analysis_20261030'
COLORS = {'qwen':'#2463a6','internvl':'#c05c20','llava':'#26815b'}
LABELS = {'qwen':'Qwen2.5-VL-7B','internvl':'InternVL3.5-8B','llava':'LLaVA-OneVision-7B-SI'}
LAYERS = {'qwen':28,'internvl':36,'llava':28}
HEADS = {'qwen':28,'internvl':32,'llava':28}
USED = {}

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def read(p):return json.loads(p.read_text(encoding='utf-8'))

def raw(p):
    with p.open(encoding='utf-8') as f:return [json.loads(x) for x in f if x.strip()]

def dump(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')

def verify(p,expected):
    actual=sha(p);assert actual==expected,str(p)
    USED[str(p.resolve())]=actual

def mean(v):
    a=[float(x) for x in v if x is not None and math.isfinite(float(x))]
    return sum(a)/len(a) if a else None

def ci(v):
    a=np.asarray(v,float);a=a[np.isfinite(a)]
    if not len(a):return {'mean':None,'ci_low':None,'ci_high':None,'n_pairs':0}
    rng=np.random.default_rng(20261030)
    b=a[rng.integers(0,len(a),(10000,len(a)))].mean(1)
    return dict(mean=float(a.mean()),ci_low=float(np.quantile(b,.025)),ci_high=float(np.quantile(b,.975)),n_pairs=len(a))

def cluster(rr,key):
    groups=defaultdict(list)
    for r in rr:
        if key in r and r[key] is not None and math.isfinite(float(r[key])):groups[r['pair_id']].append(float(r[key]))
    return ci([mean(v) for _,v in sorted(groups.items())])

def csvout(name,rr):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,sorted(set().union(*(r.keys() for r in rr))),lineterminator='\n');w.writeheader();w.writerows(rr)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--llava-root',type=Path,default=ROOT.parent/'ANLP-project-jarvis-pilots')
    a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True);(OUT/'figures').mkdir(exist_ok=True);(OUT/'evidence').mkdir(exist_ok=True)
    lp=a.llava_root/'consolidation/llava_iconqa_audit_20261030'
    lm=read(lp/'REPORT_ARTIFACT_MANIFEST.json')
    for r in lm['files']:
        if r['path'] in ['audit_summary.json','REPORT.md','evidence/final_lens_comparison.json']:
            verify(lp/r['path'],r['sha256'])
    audit=read(lp/'audit_summary.json');assert audit['audit_status']=='qualified_pass' and not audit['checks_failed']
    dump(OUT/'evidence/llava_audit_summary.json',audit)
    dump(OUT/'evidence/llava_final_lens_comparison.json',read(lp/'evidence/final_lens_comparison.json'))
    locations={j:ROOT/'results/ada_completed_run_backups'/('job_'+job) for j,job in [('qwen','46804'),('internvl','46912')]}
    locations['llava']=a.llava_root/'results/jarvis_llava_backup_20261030/portable_v2/extracted'
    sources={}
    for model,base in locations.items():
        if model=='llava':
            inv=read(base/'EXPORT_MANIFEST.json');assert sha(base/'EXPORT_MANIFEST.json')==audit['input_manifest_sha256']
            sources[model]={r['path']:(base/r['path'],r['sha256']) for r in inv['files']}
        else:
            inv=read(base/'remote_inventory.json');receipt=read(base/'transfer_complete.json')
            verify(base/'remote_inventory.json',receipt['inventory_sha256'])
            sources[model]={name:(Path(r['local_path']) if 'local_path' in r else base/name,r['sha256']) for name,r in inv['files'].items()}
    def get(model,name,jsonl=False):
        path,digest=sources[model][name];verify(path,digest)
        return raw(path) if jsonl else read(path)
    result={'identity':'ICONQA-THREE-MODEL-SYNTHESIS-20261030','scope':'CF-small only; baseline-conditioned reduced exploratory causal study',
            'statistics':'Equal source-pair means; pointwise percentile bootstrap, 10000 draws, NumPy seed 20261030. Frozen decisions unchanged.',
            'models':{},'paired_behavior':{},'cohort_overlap':{}}
    behavior_table=[];causal_table=[];sensitivity=[];head_table=[];lens_table=[];projection_table=[];control_table=[];ablation_table=[]
    behavior_raw={};selections={};head_raw={}
    for model,base in locations.items():
        items={r['item_id']:r for r in get(model,'dataset/items.jsonl',True)}
        release=get(model,'dataset/release_manifest.json')
        assert release['release_sha256']=='93efc68344571dda71999086df9b4638294220bce19901eb11998f519931c87c'
        identity=get(model,'results/resume_identity.json');assert identity['seed']==20261029
        freeze=get(model,'results/iconqa_freeze.json')
        source_path,source_digest=sources[model]['source/src/vlm_mcqa/core7/components.py']
        verify(source_path,source_digest)
        component_source=source_path.read_text(encoding='utf-8')
        assert 'donor = resample_donor' in component_source and 'donor_clean.sites' in component_source
        entry={'model':LABELS[model],'layers':LAYERS[model],'heads_per_layer':HEADS[model],
               'dataset_hash':freeze['dataset_hash'],'release_sha256':release['release_sha256'],
               'batch_size':identity['batch_size'],'source_identity':identity.get('source_manifest',identity.get('source')),
               'ablation_definition':'Activation replacement from a different clean example with a different correct symbol; donor resampling, not mean or zero ablation.',
               'freeze':freeze,'phases':{}}
        result['models'][model]=entry;behavior_raw[model]={};selections[model]={};head_raw[model]={}
        for phase in ['discovery','confirmation']:
            prefix='results/'+phase+'/'
            def load(name,jsonl=False):return get(model,prefix+name,jsonl)
            b=load('behavior_'+phase+'_summary.json');elig=load('eligibility.json');selected=set(elig['pair_ids']);selections[model][phase]=selected
            agg=load('aggregate_'+phase+'.json');cal=load('calibration.json');assert cal['all_passed']
            rr=load('behavior_'+phase+'.jsonl',True)
            for r in rr:
                item=items[r['item_id']];r['image_status']=item['image_status'];r['semantic_answer']=item['semantic_answer']
            behavior_raw[model][phase]=rr
            ph={'behavior':b,'selection_scanned':elig['scanned_pairs'],'selected_pairs':len(selected),
                'aggregate':{k:v for k,v in agg.items() if k!='heads'},'behavior_strata':{},'head_arms':{}}
            entry['phases'][phase]=ph
            for scheme in ['letters','numbers','rare_letters']:
                ss=[r for r in rr if r['scheme']==scheme and r['condition']=='clean'];assert len(ss)==512
                for stratum,subset in [('all',ss)]+[(status,[r for r in ss if r['image_status']==status]) for status in ['native','edited']]:
                    stat=cluster(subset,'correct');ph['behavior_strata'][scheme+'/'+stratum]=stat
                    behavior_table.append(dict(model=model,phase=phase,scheme=scheme,stratum=stratum,**stat))
                classes={c:mean(r['correct'] for r in ss if r['semantic_answer']==c) for c in ['certain','probable','impossible','unlikely']}
                ph['behavior_strata'][scheme+'/classes']=classes;ph['behavior_strata'][scheme+'/macro_accuracy']=mean(classes.values())
            patch=load('patching_'+phase+'.jsonl',True)
            for site_file in [('patching',patch),('components',load('components_'+phase+'.jsonl',True))]:
                kindfile,data=site_file;groups=defaultdict(list)
                for r in data:
                    if r['contrast_type']=='identity':continue
                    groups[(r['contrast_type'],r['site'],r.get('group','final'),r['layer'])].append(r)
                for (kind,site,group,layer),subset in sorted(groups.items()):
                    if kindfile=='components' and kind!='position':continue
                    causal_table.append(dict(model=model,phase=phase,contrast=kind,site=site,group=group,layer=layer,
                                             depth=(layer+1)/LAYERS[model],**cluster(subset,'normalized_recovery')))
            content=[r for r in patch if r['contrast_type']=='content' and r['group']=='final']
            unique={r['contrast_id']:r for r in content};assert len(unique)==128
            den=[r['donor_diff']-r['recipient_diff'] for r in unique.values()]
            diagnostic={'contrasts':128,'negative':sum(x<0 for x in den),'abs_below_01':sum(abs(x)<.1 for x in den),
                        'abs_below_1':sum(abs(x)<1 for x in den),'min_abs':min(map(abs,den)),'crossings':{}}
            for layer in range(LAYERS[model]):
                sub=[r for r in content if r['layer']==layer];strong=[r for r in sub if abs(r['donor_diff']-r['recipient_diff'])>=1]
                estimates={'mean_ratios':mean(r['normalized_recovery'] for r in sub),
                           'median_ratios':float(np.median([r['normalized_recovery'] for r in sub])),
                           'ratio_means':mean(r['patched_diff']-r['recipient_diff'] for r in sub)/mean(r['donor_diff']-r['recipient_diff'] for r in sub),
                           'abs_den_ge_1':mean(r['normalized_recovery'] for r in strong)}
                for estimator,value in estimates.items():
                    sensitivity.append(dict(model=model,phase=phase,layer=layer,estimator=estimator,value=value,n_contrasts=len(strong) if estimator=='abs_den_ge_1' else len(sub)))
                    if value is not None and value>=.5 and estimator not in diagnostic['crossings']:diagnostic['crossings'][estimator]=layer
            ph['content_normalization']=diagnostic
            for arm,folder in [('iconqa',''),('clevr','clevr_frozen_heads/')]:
                summary=load(folder+'heads_'+phase+'_summary.json');head_records=load(folder+'heads_validate_'+phase+'.jsonl',True)
                head_raw[model].setdefault(phase,{})[arm]=head_records
                ref=summary['all_heads_mean_nr'];ph['head_arms'][arm]={'all_heads_mean_nr':ref,'reference_head_count':len((freeze if arm=='iconqa' else load(folder+'source_freeze.json'))['head_layers'])*HEADS[model],
                   'candidates':summary['candidates'],'verdict':summary['verdict'],'per_k':summary['per_k']}
                for k,v in summary['per_k'].items():
                    sub=[r for r in head_records if r['set']=='top'+k]
                    head_table.append(dict(model=model,phase=phase,arm=arm,k=int(k),all_head_reference=ref,fraction_all=v['fraction_of_all_heads'],random_p95=v['random_p95'],**cluster(sub,'normalized_recovery')))
                aa=load(folder+'heads_ablation_'+phase+'.jsonl',True)
                for r in aa:r['pair_id']=items[r['recipient_variant_id'].split('|')[0]]['counterfactual_pair_id']
                for name in ['top4','top8','all_heads','random4_0']:
                    subset=[r for r in aa if r['head_set']==name]
                    if subset:ablation_table.append(dict(model=model,phase=phase,arm=arm,head_set=name,clean_margin=mean(r['clean_margin'] for r in subset),still_correct=mean(r['still_correct'] for r in subset),**cluster(subset,'margin_drop')))
            cc=load('edit_controls_'+phase+'.jsonl',True);assert len(cc)==32
            counts={len(r['frozen_heads']) for r in cc};assert len(counts)==1
            for condition in ['counterfactual','same_answer','paired_difference']:
                vals=[r['conditions']['counterfactual']['patched_minus_native']-r['conditions']['same_answer']['patched_minus_native'] if condition=='paired_difference' else r['conditions'][condition]['patched_minus_native'] for r in cc]
                control_table.append(dict(model=model,phase=phase,condition=condition,head_count=next(iter(counts)),**ci(vals)))
            ph['edit_control_summary']=load('edit_controls_'+phase+'_summary.json')
            ph['control_head_count']=next(iter(counts))
            groups=defaultdict(list)
            for r in load('lens_'+phase+'.jsonl',True):groups[(r['scheme'],r['stage'])].append(r)
            for (scheme,stage),subset in sorted(groups.items()):
                stat={k:mean(r.get(k) for r in subset) for k in ['mean_label_logit','margin','label_mass_full','correct','correct_restricted_prob','requested_mass_8way','content_margin']}
                lens_table.append(dict(model=model,phase=phase,scheme=scheme,stage=stage,depth=stage/LAYERS[model],**stat))
            groups=defaultdict(list)
            for r in load('component_projection_'+phase+'.jsonl',True):
                if r['scheme']=='letters':groups[(r['component'],r['layer'])].append(r)
            for (component,layer),subset in sorted(groups.items()):
                projection_table.append(dict(model=model,phase=phase,component=component,layer=layer,
                   common=mean(r['direct_mean_label'] for r in subset),separation=mean(r['direct_correct_minus_best_incorrect'] for r in subset)))
            if phase=='confirmation':ph['replication_checks']=load('confirmation_report.json')['checks']
            print('Synthesized',model,phase,flush=True)
    for phase in ['discovery','confirmation']:
        result['cohort_overlap'][phase]={'all_three':len(set.intersection(*(selections[m][phase] for m in LABELS)))}
        result['paired_behavior'][phase]={}
        for left,right in itertools.combinations(LABELS,2):
            shared=selections[left][phase]&selections[right][phase]
            result['cohort_overlap'][phase][left+'/'+right]=len(shared)
            for scheme in ['letters','numbers','rare_letters']:
                l={r['variant_id']:r for r in behavior_raw[left][phase] if r['condition']=='clean' and r['scheme']==scheme}
                r={x['variant_id']:x for x in behavior_raw[right][phase] if x['condition']=='clean' and x['scheme']==scheme}
                assert l.keys()==r.keys()
                diff=[dict(pair_id=l[k]['pair_id'],delta=int(l[k]['correct'])-int(r[k]['correct'])) for k in l]
                result['paired_behavior'][phase][left+'-'+right+'/'+scheme]={**cluster(diff,'delta'),
                    'left_only_correct':sum(l[k]['correct'] and not r[k]['correct'] for k in l),
                    'right_only_correct':sum(r[k]['correct'] and not l[k]['correct'] for k in l)}
    # Common-eligible-pair sensitivity uses each model's saved head cases. The
    # selected native/edited member/layout can differ, so this is not exact-case matching.
    result['common_pair_head_sensitivity']={}
    for phase in ['discovery','confirmation']:
        shared=set.intersection(*(selections[m][phase] for m in LABELS))
        result['common_pair_head_sensitivity'][phase]={m:cluster([r for r in head_raw[m][phase]['iconqa'] if r['set']=='top4' and r['pair_id'] in shared],'normalized_recovery') for m in LABELS}
    for name,data in [('behavior.csv',behavior_table),('causal_curves.csv',causal_table),('content_sensitivity.csv',sensitivity),
                      ('heads.csv',head_table),('fixed_text_controls.csv',control_table),('ablation.csv',ablation_table),
                      ('lens.csv',lens_table),('projections.csv',projection_table)]:csvout(name,data)
    dump(OUT/'summary.json',result)
    plot(behavior_table,causal_table,head_table,lens_table,projection_table,sensitivity)
    dump(OUT/'input_provenance.json',{'analysis_script_sha256':sha(Path(__file__)),
        'verified_inputs':USED,'llava_audit_checks':audit['checks_total'],'note':'Reuses prior audits; verifies hashes of inputs used here. No new inference.'})
    refresh_manifest()
    print('Combined analysis complete:',OUT,flush=True)

def plot(behavior,curves,heads,lens,proj,sensitivity):
    plt.rcParams.update({'font.size':10,'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for n,m in enumerate(LABELS):
        for ax,strata,schemes in [(axs[0],['all']*3,['letters','numbers','rare_letters']),
                                 (axs[1],['native','edited'],['letters']*2)]:
            rr=[next(r for r in behavior if r['model']==m and r['phase']=='confirmation' and r['scheme']==s and r['stratum']==st) for s,st in zip(schemes,strata)]
            x=np.arange(len(rr))+(n-1)*.24;y=np.array([r['mean'] for r in rr]);lo=[r['mean']-r['ci_low'] for r in rr];hi=[r['ci_high']-r['mean'] for r in rr]
            ax.bar(x,y,.23,label=LABELS[m],color=COLORS[m],yerr=[lo,hi],capsize=3)
    axs[0].set(xticks=[0,1,2],xticklabels=['A/B/C/D','Numbers','Rare letters'],ylim=(0,1.06),ylabel='Accuracy',title='Confirmation: same 128 images, 64 source pairs')
    axs[1].set(xticks=[0,1],xticklabels=['Native','Edited'],ylim=(0,1.06),title='A/B/C/D by image status');axs[0].legend(fontsize=8)
    save(fig,'01_behavior')
    fig,axs=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for ax,m in zip(axs,LABELS):
        for kind,style in [('content','-'),('position','--')]:
            rr=[r for r in curves if r['model']==m and r['phase']=='confirmation' and r['site']=='resid_post' and r['group']=='final' and r['contrast']==kind]
            ax.plot([r['depth'] for r in rr],[r['mean'] for r in rr],style,color=COLORS[m],label=kind,lw=2)
            ax.fill_between([r['depth'] for r in rr],[r['ci_low'] for r in rr],[r['ci_high'] for r in rr],color=COLORS[m],alpha=.13)
        ax.axhline(.5,color='gray',lw=.8);ax.set(title=LABELS[m],xlabel='Fraction of decoder depth',ylabel='Normalized recovery');ax.legend()
    fig.suptitle('Content and position transfer: magnitude depends on readout denominator')
    save(fig,'02_residual_transfer')
    fig,axs=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for ax,m in zip(axs,LABELS):
        for site,style in [('attn','-'),('mlp','--')]:
            rr=[r for r in curves if r['model']==m and r['phase']=='confirmation' and r['site']==site and r['contrast']=='position']
            ax.plot([r['depth'] for r in rr],[r['mean'] for r in rr],style,color=COLORS[m],label=site,lw=2)
        ax.set(title=LABELS[m],xlabel='Fraction of decoder depth',ylabel='Position recovery');ax.legend()
    save(fig,'03_components')
    fig,axs=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for m in LABELS:
        for ax,arm in zip(axs,['iconqa','clevr']):
            rr=sorted([r for r in heads if r['model']==m and r['phase']=='confirmation' and r['arm']==arm],key=lambda r:r['k'])
            ax.plot([r['k'] for r in rr],[r['fraction_all'] for r in rr],'-o',label=LABELS[m],color=COLORS[m])
            ax.axhline(.5,color='gray',lw=.5)
            ax.set(xlabel='Number of patched heads',ylabel='Fraction of selected-layer all-head effect',xticks=[1,2,4,8,16])
    axs[0].set_title('IconQA-selected, frozen before confirmation');axs[1].set_title('Historical CLEVR-frozen candidates');axs[0].legend(fontsize=8)
    save(fig,'04_heads')
    fig,axs=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for ax,m in zip(axs,LABELS):
        for group,color in [('image','#2463a6'),('options','#c05c20'),('labels','#26815b'),('question','#8c5a9f')]:
            rr=[r for r in curves if r['model']==m and r['phase']=='confirmation' and r['site']=='resid_post' and r['group']==group and r['contrast']=='content_and_position']
            ax.plot([r['depth'] for r in rr],[r['mean'] for r in rr],label=group,color=color)
        ax.set(title=LABELS[m],xlabel='Fraction of decoder depth',ylabel='Factorial normalized recovery');ax.legend(fontsize=8)
    save(fig,'05_token_groups')
    fig,axs=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
    for n,m in enumerate(LABELS):
        rr=[r for r in lens if r['model']==m and r['phase']=='confirmation' and r['scheme']=='letters']
        axs[0,n].plot([r['depth'] for r in rr],[r['label_mass_full'] for r in rr],color=COLORS[m]);axs[0,n].set(title=LABELS[m],ylabel='Full-vocabulary mass on A/B/C/D',ylim=(0,1.05),xlabel='Fraction of residual stages')
        for component,style in [('attn','-'),('mlp','--')]:
            pp=[r for r in proj if r['model']==m and r['phase']=='confirmation' and r['component']==component]
            axs[1,n].plot([(r['layer']+1)/LAYERS[m] for r in pp],[r['separation'] for r in pp],style,label=component,color=COLORS[m])
        axs[1,n].set(xlabel='Fraction of decoder depth',ylabel='Direct correct-label separation');axs[1,n].legend()
    fig.suptitle('Observational readouts: format mass and component separation are distinct from causality')
    save(fig,'06_readouts')
    fig,axs=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for ax,m in zip(axs,LABELS):
        for estimator,style in [('mean_ratios','-'),('median_ratios','--'),('ratio_means',':'),('abs_den_ge_1','-.')]:
            rr=[r for r in sensitivity if r['model']==m and r['phase']=='confirmation' and r['estimator']==estimator]
            ax.plot([r['layer'] for r in rr],[r['value'] for r in rr],style,label=estimator)
        ax.axhline(.5,color='gray',lw=.8);ax.axvline(30 if m=='internvl' else 24,color='gray',lw=.8,alpha=.5)
        ax.set(title=LABELS[m],xlabel='Decoder layer (zero based)',ylabel='Content recovery diagnostic');ax.legend(fontsize=7)
    fig.suptitle('Normalization sensitivity: vertical line marks original position threshold')
    save(fig,'07_normalization_sensitivity')

def save(fig,name):
    fig.savefig(OUT/'figures'/f'{name}.png',bbox_inches='tight');plt.close(fig)

def refresh_manifest():
    dump(OUT/'MANIFEST.json',{'files':{str(f.relative_to(OUT)).replace('\\','/'):sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name!='MANIFEST.json'}})

if __name__=='__main__':main()
