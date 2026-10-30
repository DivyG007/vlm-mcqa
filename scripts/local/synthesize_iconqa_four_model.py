"""Extend immutable three-model IconQA synthesis with audited exploratory Pali.

Pali uses expanded stimuli/definition prompts/BF16 and amended behavior admission.
It is not silently pooled into the matched three-model accuracy comparison.
"""
from __future__ import annotations
import argparse
import csv
import itertools
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import synthesize_iconqa_three_model as core

ROOT=Path(__file__).resolve().parents[2]
PRIOR=ROOT/'consolidation/iconqa_three_model_analysis_20261030'
OUT=ROOT/'consolidation/iconqa_four_model_analysis_20261030'
LABELS={**core.LABELS,'pali':'PaliGemma2-10B (extension)'}
COLORS={**core.COLORS,'pali':'#8658a5'}
LAYERS={**core.LAYERS,'pali':42}
read,raw,sha,ci,cluster,mean,dump=core.read,core.raw,core.sha,core.ci,core.cluster,core.mean,core.dump

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jarvis-root',type=Path,default=ROOT.parent/'ANLP-project-jarvis-pilots')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True);(OUT/'figures').mkdir(exist_ok=True);(OUT/'evidence').mkdir(exist_ok=True)
    core.OUT=OUT
    prior_manifest=read(PRIOR/'MANIFEST.json')
    for name,digest in prior_manifest['files'].items():core.verify(PRIOR/name,digest)
    summary=read(PRIOR/'summary.json')
    summary.update(identity='ICONQA-FOUR-MODEL-SYNTHESIS-20261030',scope='Three matched CF-small models plus separately identified expanded/definition-prompt Pali exploratory extension.')
    summary['comparison_policy']={'matched_release':['qwen','internvl','llava'],'exploratory_extension':['pali'],
        'pali_accuracy_ranking':'No unchanged-protocol model ranking: different release, prompt, precision and admission.',
        'statistics':'Source-pair means; 10000 pointwise percentile-bootstrap draws; NumPy seed 20261030. No multiplicity correction.'}
    tables={name:list(csv.DictReader((PRIOR/(name+'.csv')).open(encoding='utf-8'))) for name in
            ['behavior','causal_curves','content_sensitivity','heads','ablation','fixed_text_controls','lens','projections']}
    for src in (PRIOR/'evidence').glob('*.json'):dump(OUT/'evidence'/src.name,read(src))
    audit_dir=args.jarvis_root/'consolidation/pali_iconqa_audit_20261030'
    audit_manifest=read(audit_dir/'REPORT_ARTIFACT_MANIFEST.json')
    for entry in audit_manifest['files']:
        if entry['path'] in ['audit_summary.json','REPORT.md','evidence/final_lens_comparison.json']:
            core.verify(audit_dir/entry['path'],entry['sha256'])
    audit=read(audit_dir/'audit_summary.json');assert audit['audit_status']=='qualified_pass' and not audit['checks_failed']
    dump(OUT/'evidence/pali_audit_summary.json',audit)
    dump(OUT/'evidence/pali_final_lens_comparison.json',read(audit_dir/'evidence/final_lens_comparison.json'))
    base=args.jarvis_root/'results/jarvis_pali_backup_20261030/portable_v1/extracted'
    core.verify(base/'EXPORT_MANIFEST.json',audit['input_manifest_sha256'])
    files={r['path']:r for r in read(base/'EXPORT_MANIFEST.json')['files']}
    def get(name,jsonl=False):
        core.verify(base/name,files[name]['sha256'])
        return raw(base/name) if jsonl else read(base/name)
    items={r['item_id']:r for r in get('dataset/items.jsonl',True)}
    variants={r['variant_id']:r for r in get('dataset/variants.jsonl',True)}
    identity=get('results/resume_identity.json');release=get('dataset/release_manifest.json');freeze=get('results/iconqa_freeze.json')
    assert release['release_sha256']=='76c470e52cba1d55aa3af76baedf9cd402f495a49399f628dfbd68d0c2610cf8'
    assert freeze['dataset_hash']=='1912e2c413c88764' and identity['seed']==20261029
    core.verify(base/'source/src/vlm_mcqa/core7/components.py',files['source/src/vlm_mcqa/core7/components.py']['sha256'])
    text=(base/'source/src/vlm_mcqa/core7/components.py').read_text(encoding='utf-8')
    assert 'donor = resample_donor' in text and 'donor_clean.sites' in text
    pali={'model':LABELS['pali'],'layers':42,'heads_per_layer':16,'batch_size':identity['batch_size'],
          'dataset_hash':freeze['dataset_hash'],'release_sha256':release['release_sha256'],'freeze':freeze,
          'source_identity':identity['source_manifest'],'train_policy':identity['train_policy'],
          'amendment':{k:identity[k] for k in ['exploratory_amendment','screen_eligible','scientific_metric_equivalence','exact_prediction_equivalence','known_pilot_failures']},
          'ablation_definition':'Different-clean-donor activation resampling, not mean/zero ablation.','phases':{}}
    summary['models']['pali']=pali
    pali_behavior={};pali_selected={}
    for phase in ['discovery','confirmation']:
        def load(name,jsonl=False):return get('results/'+phase+'/'+name,jsonl)
        behavior=load('behavior_'+phase+'_summary.json');elig=load('eligibility.json');agg=load('aggregate_'+phase+'.json')
        assert load('calibration.json')['all_passed'] and elig['pairs_used']==32
        ph={'behavior':behavior,'selected_pairs':32,'selection_scanned':elig['scanned_pairs'],
            'aggregate':{k:v for k,v in agg.items() if k!='heads'},'behavior_admission':load('behavior_admission.json'),
            'behavior_strata':{},'head_arms':{}}
        pali['phases'][phase]=ph;pali_selected[phase]=set(elig['pair_ids'])
        rr=load('behavior_'+phase+'.jsonl',True);pali_behavior[phase]=rr
        for r in rr:r.update(image_status=items[r['item_id']]['image_status'],semantic_answer=items[r['item_id']]['semantic_answer'])
        for scheme in ['letters','numbers','rare_letters']:
            ss=[r for r in rr if r['condition']=='clean' and r['scheme']==scheme];assert len(ss)==512 and len({r['pair_id'] for r in ss})==64
            for stratum,subset in [('all',ss)]+[(s,[r for r in ss if r['image_status']==s]) for s in ['native','edited']]:
                stat=cluster(subset,'correct');ph['behavior_strata'][scheme+'/'+stratum]=stat
                tables['behavior'].append(dict(model='pali',phase=phase,scheme=scheme,stratum=stratum,**stat))
            classes={c:mean(r['correct'] for r in ss if r['semantic_answer']==c) for c in ['certain','probable','impossible','unlikely']}
            ph['behavior_strata'][scheme+'/classes']=classes;ph['behavior_strata'][scheme+'/macro_accuracy']=mean(classes.values())
        patch=load('patching_'+phase+'.jsonl',True)
        for stem,data in [('patching',patch),('components',load('components_'+phase+'.jsonl',True))]:
            grouped={}
            for r in data:
                if r['contrast_type']=='identity' or (stem=='components' and r['contrast_type']!='position'):continue
                grouped.setdefault((r['contrast_type'],r['site'],r.get('group','final'),r['layer']),[]).append(r)
            for (kind,site,group,layer),sub in sorted(grouped.items()):
                tables['causal_curves'].append(dict(model='pali',phase=phase,contrast=kind,site=site,group=group,layer=layer,depth=(layer+1)/42,**cluster(sub,'normalized_recovery')))
        content=[r for r in patch if r['contrast_type']=='content' and r['group']=='final'];unique={r['contrast_id']:r for r in content};assert len(unique)==128
        den=[r['donor_diff']-r['recipient_diff'] for r in unique.values()]
        diag={'contrasts':128,'negative':sum(x<0 for x in den),'abs_below_01':sum(abs(x)<.1 for x in den),'abs_below_1':sum(abs(x)<1 for x in den),'min_abs':min(map(abs,den)),'max_abs':max(map(abs,den)),'crossings':{}}
        for layer in range(42):
            sub=[r for r in content if r['layer']==layer]
            estimates={'mean_ratios':mean(r['normalized_recovery'] for r in sub),'median_ratios':float(np.median([r['normalized_recovery'] for r in sub])),
                       'ratio_means':mean(r['patched_diff']-r['recipient_diff'] for r in sub)/mean(r['donor_diff']-r['recipient_diff'] for r in sub)}
            for threshold,key in [(1,'abs_den_ge_1'),(.1,'abs_den_ge_01')]:estimates[key]=mean(r['normalized_recovery'] for r in sub if abs(r['donor_diff']-r['recipient_diff'])>=threshold)
            for key,value in estimates.items():
                tables['content_sensitivity'].append(dict(model='pali',phase=phase,layer=layer,estimator=key,value=value,n_contrasts=sum(abs(r['donor_diff']-r['recipient_diff'])>=(1 if key=='abs_den_ge_1' else .1) for r in sub) if key.startswith('abs_den') else 128))
                if value is not None and value>=.5 and key not in diag['crossings']:diag['crossings'][key]=layer
        ph['content_normalization']=diag
        for arm,folder in [('iconqa',''),('clevr','clevr_frozen_heads/')]:
            h=load(folder+'heads_'+phase+'_summary.json');hr=load(folder+'heads_validate_'+phase+'.jsonl',True)
            ph['head_arms'][arm]={'all_heads_mean_nr':h['all_heads_mean_nr'],'reference_head_count':64,'candidates':h['candidates'],'per_k':h['per_k'],'verdict':h['verdict']}
            for k,v in h['per_k'].items():tables['heads'].append(dict(model='pali',phase=phase,arm=arm,k=int(k),all_head_reference=h['all_heads_mean_nr'],fraction_all=v['fraction_of_all_heads'],random_p95=v['random_p95'],**cluster([r for r in hr if r['set']=='top'+k],'normalized_recovery')))
            aa=load(folder+'heads_ablation_'+phase+'.jsonl',True)
            for r in aa:r['pair_id']=items[r['recipient_variant_id'].split('|')[0]]['counterfactual_pair_id']
            for name in ['top2','top4','top8','all_heads','random4_0']:
                sub=[r for r in aa if r['head_set']==name]
                if sub:tables['ablation'].append(dict(model='pali',phase=phase,arm=arm,head_set=name,clean_margin=mean(r['clean_margin'] for r in sub),still_correct=mean(r['still_correct'] for r in sub),**cluster(sub,'margin_drop')))
        controls=load('edit_controls_'+phase+'.jsonl',True);assert len(controls)==32
        for condition in ['counterfactual','same_answer','paired_difference']:
            values=[r['conditions']['counterfactual']['patched_minus_native']-r['conditions']['same_answer']['patched_minus_native'] if condition=='paired_difference' else r['conditions'][condition]['patched_minus_native'] for r in controls]
            tables['fixed_text_controls'].append(dict(model='pali',phase=phase,condition=condition,head_count=len(controls[0]['frozen_heads']),**ci(values)))
        ph['edit_control_summary']=load('edit_controls_'+phase+'_summary.json');ph['control_head_count']=len(controls[0]['frozen_heads'])
        groups={}
        for r in load('lens_'+phase+'.jsonl',True):groups.setdefault((r['scheme'],r['stage']),[]).append(r)
        for (scheme,stage),sub in sorted(groups.items()):tables['lens'].append(dict(model='pali',phase=phase,scheme=scheme,stage=stage,depth=stage/42,**{k:mean(r.get(k) for r in sub) for k in ['mean_label_logit','margin','label_mass_full','correct','correct_restricted_prob','requested_mass_8way','content_margin']}))
        groups={}
        for r in load('component_projection_'+phase+'.jsonl',True):
            if r['scheme']=='letters':groups.setdefault((r['component'],r['layer']),[]).append(r)
        for (component,layer),sub in sorted(groups.items()):tables['projections'].append(dict(model='pali',phase=phase,component=component,layer=layer,common=mean(r['direct_mean_label'] for r in sub),separation=mean(r['direct_correct_minus_best_incorrect'] for r in sub)))
        if phase=='confirmation':ph['replication_checks']=load('confirmation_report.json')['checks']
        print('Integrated exploratory Pali',phase,flush=True)
    # Match source images and displayed option mappings, without equating the
    # different actual definition prompt or precision. Report no paired Pali p-value.
    qb=ROOT/'results/ada_completed_run_backups/job_46804';qi=read(qb/'remote_inventory.json')
    def qget(name,jsonl=False):
        entry=qi['files'][name];path=Path(entry['local_path']) if 'local_path' in entry else qb/name;core.verify(path,entry['sha256'])
        return raw(path) if jsonl else read(path)
    orig_items={r['item_id']:r for r in qget('dataset/items.jsonl',True)};orig_variants={r['variant_id']:r for r in qget('dataset/variants.jsonl',True)}
    summary['pali_common_stimuli']={}
    other_sources={}
    for model,root,manifest_name in [('internvl',ROOT/'results/ada_completed_run_backups/job_46912','remote_inventory.json'),
                                     ('llava',args.jarvis_root/'results/jarvis_llava_backup_20261030/portable_v2/extracted','EXPORT_MANIFEST.json')]:
        mm=read(root/manifest_name)
        other_sources[model]={name:(Path(r['local_path']) if 'local_path'in r else root/name,r['sha256']) for name,r in mm['files'].items()} if model=='internvl' else {r['path']:(root/r['path'],r['sha256']) for r in mm['files']}
    def otherget(model,name,jsonl=False):
        path,digest=other_sources[model][name];core.verify(path,digest)
        return raw(path) if jsonl else read(path)
    for phase in ['discovery','confirmation']:
        pb={r['variant_id']:r for r in pali_behavior[phase] if r['condition']=='clean' and r['scheme']=='letters'}
        qbrows={r['variant_id']:r for r in qget('results/'+phase+'/behavior_'+phase+'.jsonl',True) if r['condition']=='clean' and r['scheme']=='letters'}
        shared=sorted(pb.keys()&qbrows.keys());assert shared
        for key in shared:
            assert orig_variants[key]['labels']==variants[key]['labels'] and orig_variants[key]['option_contents']==variants[key]['option_contents']
            item=orig_items[pb[key]['item_id']];name='dataset/'+item['image'];entry=qi['files'][name];oldpath=Path(entry['local_path']) if 'local_path'in entry else qb/name
            newname='dataset/'+items[item['item_id']]['image'];core.verify(oldpath,entry['sha256']);core.verify(base/newname,files[newname]['sha256'])
            assert sha(oldpath)==sha(base/newname),'common image bytes differ'
        selected=qget('results/'+phase+'/eligibility.json')['pair_ids']
        common_accuracy={'qwen':cluster([qbrows[k] for k in shared],'correct'),'pali':cluster([pb[k] for k in shared],'correct')}
        selected_sets={'qwen':set(selected),'pali':pali_selected[phase]}
        for model in ['internvl','llava']:
            other={r['variant_id']:r for r in otherget(model,'results/'+phase+'/behavior_'+phase+'.jsonl',True) if r['condition']=='clean' and r['scheme']=='letters'}
            assert all(k in other for k in shared)
            common_accuracy[model]=cluster([other[k] for k in shared],'correct')
            selected_sets[model]=set(otherget(model,'results/'+phase+'/eligibility.json')['pair_ids'])
        summary['pali_common_stimuli'][phase]={'letter_variants':len(shared),'source_pairs':len({pb[k]['pair_id'] for k in shared}),
            'qwen_accuracy':cluster([qbrows[k] for k in shared],'correct'),'pali_accuracy':cluster([pb[k] for k in shared],'correct'),
            'accuracy_by_executed_condition':common_accuracy,'causal_overlap_all_four':len(set.intersection(*selected_sets.values())),
            'causal_overlap_with_pali':{model:len(ids&pali_selected[phase]) for model,ids in selected_sets.items() if model!='pali'},
            'note':'Identical image bytes/option mapping, different executed prompt and precision; descriptive protocol comparison, not model-only effect.',
            'causal_pair_overlap_qwen':len(set(selected)&pali_selected[phase])}
    for name,data in tables.items():core.csvout(name+'.csv',data)
    dump(OUT/'summary.json',summary)
    plot(tables,summary)
    dump(OUT/'input_provenance.json',{'synthesis_source_sha256':sha(Path(__file__)),'helper_source_sha256':sha(ROOT/'scripts/local/synthesize_iconqa_three_model.py'),
        'prior_three_model_manifest_sha256':sha(PRIOR/'MANIFEST.json'),'verified_inputs':core.USED,
        'prior_pali_audit_checks':audit['checks_total'],'scope':'Reuses prior complete audits; hashes inputs used here. No model inference or claim that all prior checks were rerun.'})
    refresh();print('Four-model report evidence complete:',OUT,flush=True)

def plot(tables,summary):
    plt.rcParams.update({'font.size':9,'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
    for name,kind in [('01_behavior','behavior'),('02_residual_transfer','residual'),('03_components','components'),('05_token_groups','groups'),('07_normalization_sensitivity','sensitivity')]:
        fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
        for ax,(m,label) in zip(axs.flat,LABELS.items()):
            if kind=='behavior':
                rr=[next(r for r in tables['behavior'] if r['model']==m and r['phase']=='confirmation' and r['scheme']==s and r['stratum']=='all') for s in ['letters','numbers','rare_letters']]
                y=np.array([float(r['mean']) for r in rr]);ax.bar(range(3),y,color=COLORS[m],yerr=[y-[float(r['ci_low']) for r in rr],[float(r['ci_high']) for r in rr]-y],capsize=3)
                ax.set(xticks=[0,1,2],xticklabels=['A/B/C/D','Numbers','Rare letters'],ylim=(0,1.05),ylabel='Accuracy on own behavior cohort')
            elif kind=='sensitivity':
                options=[('mean_ratios','Mean of ratios'),('median_ratios','Median ratio'),('ratio_means','Ratio of means'),('abs_den_ge_01' if m=='pali' else 'abs_den_ge_1','|denominator| ≥0.1' if m=='pali' else '|denominator| ≥1')]
                for key,desc in options:
                    rr=[r for r in tables['content_sensitivity'] if r['model']==m and r['phase']=='confirmation' and r['estimator']==key]
                    ax.plot([int(r['layer']) for r in rr],[float(r['value']) if r['value'] not in ['',None] else np.nan for r in rr],label=desc)
                ax.axhline(.5,color='gray',lw=.7);ax.axvline(summary['models'][m]['phases']['confirmation']['aggregate']['position_transfer_layer'],color='gray',lw=.7)
                ax.set(xlabel='Decoder layer (zero based)',ylabel='Content recovery diagnostic');ax.legend(fontsize=7)
            else:
                specs={'residual':[('content','final','resid_post','-'),('position','final','resid_post','--')],
                       'components':[('position','final','attn','-'),('position','final','mlp','--')],
                       'groups':[('content_and_position',g,'resid_post','-') for g in ['image','options','question','labels']]}[kind]
                for n,(contrast,group,site,style) in enumerate(specs):
                    rr=[r for r in tables['causal_curves'] if r['model']==m and r['phase']=='confirmation' and r['contrast']==contrast and r['group']==group and r['site']==site]
                    xx=[float(r['depth']) for r in rr];yy=[float(r['mean']) for r in rr]
                    ax.plot(xx,yy,style,label=contrast if kind=='residual' else site if kind=='components' else group,color=COLORS[m] if kind!='groups' else None)
                    if kind=='residual':ax.fill_between(xx,[float(r['ci_low']) for r in rr],[float(r['ci_high']) for r in rr],alpha=.12,color=COLORS[m])
                if kind=='residual':ax.axhline(.5,color='gray',lw=.7)
                ax.set(xlabel='Fraction of decoder depth',ylabel='Normalized recovery');ax.legend(fontsize=8)
            ax.set_title(label)
        fig.suptitle('Confirmation: Pali uses expanded stimuli, definitions and BF16; not an unchanged-protocol ranking')
        fig.savefig(OUT/'figures'/f'{name}.png',bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for m,label in LABELS.items():
        for ax,arm in zip(axs,['iconqa','clevr']):
            rr=sorted([r for r in tables['heads'] if r['model']==m and r['phase']=='confirmation' and r['arm']==arm],key=lambda r:int(r['k']))
            ax.plot([int(r['k']) for r in rr],[float(r['fraction_all']) for r in rr],'-o',color=COLORS[m],label=label)
            ax.axhline(.5,color='gray',lw=.6);ax.set(xlabel='Number of heads patched',ylabel='Fraction of own selected-layer reference',xticks=[1,2,4,8,16],title='IconQA-selected' if arm=='iconqa' else 'Historical CLEVR-frozen')
    axs[0].legend(fontsize=7);fig.suptitle('Conditional concentration across distinct model cohorts and protocols')
    fig.savefig(OUT/'figures/04_heads.png',bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for i,(m,label) in enumerate(LABELS.items()):
        rr=[r for r in tables['fixed_text_controls'] if r['model']==m and r['phase']=='confirmation']
        for ax,conditions in zip(axs,[['counterfactual','same_answer'],['paired_difference']]):
            for n,condition in enumerate(conditions):
                r=next(r for r in rr if r['condition']==condition);x=i+(n-.5)*.25 if len(conditions)>1 else i
                y=float(r['mean']);ax.bar(x,y,.23,color=COLORS[m],alpha=1 if condition!='same_answer' else .4,yerr=[[y-float(r['ci_low'])],[float(r['ci_high'])-y]],capsize=3,label=condition if i==0 else None)
            ax.set(xticks=range(4),xticklabels=['Qwen\n4 heads','InternVL\n16 heads','LLaVA\n16 heads','Pali ext.\n16 heads'],ylabel='Target logit-difference change')
    axs[0].set_title('Answer-changing versus same-answer image');axs[0].legend(fontsize=8);axs[1].set_title('Within-pair answer-changing minus control')
    fig.suptitle('Fixed-text image controls; raw amplitudes are not a fair model ranking')
    fig.savefig(OUT/'figures/06_image_controls.png',bbox_inches='tight');plt.close(fig)

def refresh():dump(OUT/'MANIFEST.json',{'files':{str(f.relative_to(OUT)).replace('\\','/'):sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name!='MANIFEST.json'}})

if __name__=='__main__':main()
