"""Full IconQA discovery/freeze/confirmation, independent of pilot completion."""
import argparse
import dataclasses
import hashlib
import json
import random
import time
import importlib.metadata
import os
import uuid
from pathlib import Path

from iconqa_qwen_pilot import verify_data, align_qwen_rotary_inputs, write, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('dataset','checkpoint','output','source-root'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--batch-size',type=int,choices=(1,2,4),default=1)
    p.add_argument('--batch-evidence',type=Path)
    args=p.parse_args()
    if not args.execute or (args.output.exists() != args.resume):
        p.error('--execute and either fresh output or explicit --resume required')
    manifest,release=verify_data(args.dataset)
    checkpoint=json.loads((args.checkpoint/'verified_checkpoint.json').read_text())
    revision='cc594898137f460bfe9f0759e9844b3ce807cfb5'
    if checkpoint.get('revision')!=revision or checkpoint.get('status')!='verified':
        raise ValueError('pinned publisher-verified checkpoint required')
    batch_evidence=None
    if args.batch_size!=1:
        if args.batch_evidence is None:
            p.error('batch 2/4 requires --batch-evidence from completed train benchmark')
        report=json.loads(args.batch_evidence.read_text())
        status=json.loads((args.batch_evidence.parent/'status.json').read_text())
        if (status.get('status')!='complete' or status.get('dataset_hash')!=manifest['dataset_hash']
                or status.get('image_aware_release_hash')!=release['image_aware_release_hash']
                or status.get('model_revision')!=revision
                or status.get('checkpoint_receipt_sha256')!=digest(args.checkpoint/'verified_checkpoint.json')):
            raise ValueError('batch benchmark model/data/completion mismatch')
        records=[r for r in report['records'] if r['batch_size']==args.batch_size]
        if (len(records)!=3 or {r['scheme'] for r in records}!={'letters','numbers','rare_letters'}
                or any(r.get('status')!='measured' or not r.get('equivalence_passed')
                       or not r.get('label_predictions_match') for r in records)):
            raise ValueError('batch benchmark equivalence gate failed')
        batch_evidence={'report_sha256':digest(args.batch_evidence),
                        'status_sha256':digest(args.batch_evidence.parent/'status.json')}
    for name, record in checkpoint['files'].items():
        path=args.checkpoint/name
        if path.stat().st_size!=record['bytes'] or (not name.endswith('.safetensors') and digest(path)!=record['sha256']):
            raise ValueError('checkpoint changed: '+name)
    args.output.mkdir(parents=True,exist_ok=args.resume)
    from iconqa_checkpoints import Blocks
    source_identity={str(f.relative_to(args.source_root)):digest(f)
                     for folder in ('src','scripts/ada','configs/core7') for f in (args.source_root/folder).rglob('*')
                     if f.is_file() and '__pycache__' not in f.parts}
    identity={'source':source_identity,'release':release['image_aware_release_hash'],
              'checkpoint_receipt':digest(args.checkpoint/'verified_checkpoint.json'),'batch_size':args.batch_size,'seed':20261028,
              'runtime':{n:importlib.metadata.version(n) for n in ('torch','transformers','accelerate')}}
    identity['batch_evidence']=batch_evidence
    binding=args.output/'resume_identity.json'
    if args.resume and json.loads(binding.read_text())!=identity:
        raise ValueError('resume source/data/model/profile binding mismatch')
    write(binding,identity)
    state={'status':'starting','pilot_dependency':False,'protocol_amendment':'user explicitly starts full campaign before pilot outcome',
           'change_policy':'later changes require new preserved versioned attempts', 'dataset_hash':manifest['dataset_hash'],
           'started_unix':time.time()}
    write(args.output/'campaign_status.json',state)
    tracker=None
    try:
        from vlm_mcqa.core7.tracking import Tracker, destination
        os.environ.setdefault('CORE7_WANDB_ENTITY','models-iiit-hyderabad5465')
        os.environ.setdefault('CORE7_WANDB_PROJECT','anlp-vlm-mcqa-staging')
        entity,project=destination()
        tracking_identity=args.output/'wandb_identity.json'
        if tracking_identity.exists():
            saved_tracking=json.loads(tracking_identity.read_text())
            if (saved_tracking['entity'],saved_tracking['project'])!=(entity,project):
                raise ValueError('W&B resume destination changed')
            wandb_id=saved_tracking['id']
        else:
            wandb_id=uuid.uuid4().hex[:12]
            write(tracking_identity,{'id':wandb_id,'entity':entity,'project':project})
        tracker=Tracker.start(enabled=True,out_dir=args.output,name=args.output.name,
                              job_type='iconqa-core7',run_id=wandb_id,
                              resume='allow' if args.resume else 'never',
                              group='iconqa-qwen25vl7b-core7',tags=['iconqa','core7','qwen25vl7b'],
                              config={'identity':identity,'dataset_hash':manifest['dataset_hash'],
                                      'output_root':str(args.output),'checkpoint_root':str(args.checkpoint)})
        state['wandb_url']=tracker.url
        write(args.output/'campaign_status.json',state)
        tracker.run.summary.update({'scheduler/job_id':os.environ.get('SLURM_JOB_ID'),
                                    'scheduler/node':os.environ.get('SLURMD_NODENAME'),
                                    'resume_requested':args.resume,'status':'running'})
        tracker.log({'lifecycle/start':1,'lifecycle/resume':args.resume})
        import torch
        from vlm_mcqa.core7.adapters import ModelSpec,VLMAdapter
        from vlm_mcqa.core7.session import Session,provenance
        from vlm_mcqa.core7.calibration import run_calibration
        from vlm_mcqa.core7.contrasts import select_cases
        from vlm_mcqa.core7.behavior import run_behavior
        from vlm_mcqa.core7.lens import run_lens
        from vlm_mcqa.core7.patching import run_patching
        from vlm_mcqa.core7.components import run_components
        from vlm_mcqa.core7.heads import run_heads
        from vlm_mcqa.core7.aggregate import aggregate_run,freeze,confirm
        from vlm_mcqa.clevr_mcq4.variants import build_random_pair_contrasts
        if torch.cuda.device_count()!=2 or any(torch.cuda.get_device_capability(i)<(7,5) for i in range(2)):
            raise RuntimeError('two Turing-or-newer GPUs required')
        placement={'model.visual':0,'model.language_model.embed_tokens':0,'model.language_model.rotary_emb':0,
                   'model.language_model.norm':1,'lm_head':1}
        placement.update({f'model.language_model.layers.{i}':0 if i<=12 else 1 for i in range(28)})
        spec=ModelSpec(key='qwen25vl_7b',family='qwen_vl',hf_id=str(args.checkpoint),revision=revision,
                       params_b=8.3,lm_backbone='qwen2.5',dtype='float16',device_map=placement,
                       processor_kwargs={'min_pixels':448*448,'max_pixels':448*448})
        adapter=VLMAdapter.load(spec)
        adapter._iconqa_device_hooks=[adapter.handle.decoder.rotary_emb.register_forward_pre_hook(align_qwen_rotary_inputs,with_kwargs=True)]
        profile=json.loads((args.source_root/'configs/core7/profiles.json').read_text())['profiles']['full_core7']
        profile.update(batch_size=args.batch_size,bounded_case_preparation=True)
        class GatedSession(Session):
            def prepared(self,*a,**kw):
                prep=super().prepared(*a,**kw)
                ids=list(prep.content_ids.values())
                if None in ids or len(set(ids))!=4 or len(prep.groups.get('image',[]))!=256:
                    raise RuntimeError('semantic-token or image-grid gate failed')
                if any(not prep.groups.get(g) for g in ('question','options','labels')):
                    raise RuntimeError('unresolved required token group')
                return prep
        def session_for(phase,out):
            s=GatedSession(adapter,args.dataset,out,profile,seed=20261028,prepared_cache=2)
            s.tracker=tracker
            s.items={k:v for k,v in s.items.items() if v['split'] in ('calibration',phase)}
            s.variants={k:v for k,v in s.variants.items() if v['item_id'] in s.items}
            s.contrasts=[v for v in s.contrasts if v['split']==phase]
            return s
        calibration_session=session_for('calibration',args.output/'calibration')
        calibration=run_calibration(calibration_session)
        tracker.summary(calibration,'calibration')
        if not calibration['all_passed']:
            raise RuntimeError('real calibration failed')
        del calibration_session
        frozen=None
        for phase in ('discovery','confirmation'):
            out=args.output/phase
            s=session_for(phase,out)
            run=provenance({'model_key':'qwen25vl_7b','model_spec':dataclasses.asdict(spec),
                            'profile':profile,'profile_name':'full_core7','num_layers':adapter.num_layers,
                            'dataset_hash':manifest['dataset_hash'],'image_aware_release_hash':release['image_aware_release_hash'],
                            'split':phase,'status':'running','wandb_url':tracker.url,
                            'methods':['behavior','lens','patching','components','heads']})
            write(out/'run.json',run); write(out/'calibration.json',calibration)
            # Baseline selection only; no intervention outcomes influence this reserve.
            cases,coverage=select_cases(s,phase,('position','content','content_and_position'),max_pairs=64,limit_per_type=None)
            write(out/'eligibility.json',coverage)
            tracker.summary({'pairs_used':coverage['pairs_used']},f'{phase}/eligibility')
            if coverage['pairs_used']!=64:
                raise RuntimeError(f'{phase}: insufficient eligible pairs, need 64, have {coverage["pairs_used"]}')
            selected=set(coverage['pair_ids']); del cases
            ids={k for k,v in s.variants.items() if v.get('counterfactual_pair_id') in selected}
            s.contrasts=[c for c in s.contrasts if c['donor_variant_id'] in ids and c['recipient_variant_id'] in ids and c['contrast_type']!='random_pair']
            pair_variants={pid:[v for v in s.variants.values() if v.get('counterfactual_pair_id')==pid] for pid in selected}
            s.contrasts+=build_random_pair_contrasts(pair_variants,random.Random(20261028))
            write(out/'selected_contrasts.json',s.contrasts)
            blocks=Blocks(out/'.checkpoints',{**identity,'phase':phase,'selection':digest(out/'selected_contrasts.json')},s.rng)
            timing={}
            actions={'behavior':lambda:run_behavior(s,split=phase,n_items=256,robust_items=256,compliance_items=256,controls=True),
                     'lens':lambda:run_lens(s,split=phase,n_items=256),
                     'patching':lambda:run_patching(s,split=phase),
                     'components':lambda:run_components(s,split=phase),
                     'heads':lambda:run_heads(s,split=phase,freeze=frozen)}
            for name,action in actions.items():
                reused=(out/'.checkpoints'/(name+'.json')).exists()
                tracker.run.summary.update({'current_phase':phase,'current_method':name})
                tracker.log({f'{phase}/{name}/started':1,f'{phase}/{name}/reused_checkpoint':reused})
                def artifacts(name=name):
                    prefixes={'behavior':('behavior_','compliance_'),'lens':('lens_','component_projection_'),
                              'patching':('patching_',),'components':('components_','component_ablation_'),
                              'heads':('heads_',)}[name]
                    return [f for f in out.iterdir() if f.is_file() and f.name.startswith(prefixes)]
                start=time.monotonic(); result=blocks.run(name,action,artifacts); timing[name]=time.monotonic()-start
                write(out/(name+'_completion.json'),{'status':'complete','seconds':timing[name],'result':result})
                tracker.log(result,f'{phase}/{name}')
                tracker.summary(result,f'{phase}/{name}')
                tracker.log({f'{phase}/{name}/invocation_seconds':timing[name],
                             f'{phase}/{name}/complete':1,
                             **{f'gpu/{i}/peak_allocated_gib':torch.cuda.max_memory_allocated(i)/2**30
                                for i in range(torch.cuda.device_count())},
                             **{f'gpu/{i}/peak_reserved_gib':torch.cuda.max_memory_reserved(i)/2**30
                                for i in range(torch.cuda.device_count())}})
                if name=='behavior' and not (result.get('worst_position_accuracy',0)>0.90
                        and result.get('clean_minus_shuffled',0)>=0.30
                        and result.get('worst_scheme_position_accuracy',0)>=0.45):
                    raise RuntimeError('strict behavioral eligibility failed; preserve descriptive outputs')
                print(f'[campaign] {phase}/{name} complete',flush=True)
            # Frozen-CLEVR-head arm is separate from newly discovered IconQA heads.
            original_out=s.out_dir; s.out_dir=out/'clevr_frozen_heads'; s.out_dir.mkdir(exist_ok=True)
            clevr={'head_layers':[19,24,26,27],'head_candidates':[[26,24],[24,27],[24,26],[24,21]],'head_set_sizes':[4]}
            write(s.out_dir/'source_freeze.json',{'clevr_freeze_hash':'4f991401f5cb7239',**clevr})
            tracker.run.summary.update({'current_phase':phase,'current_method':'clevr_transfer'})
            transfer=blocks.run('clevr_transfer',lambda:run_heads(s,split=phase,freeze=clevr),
                       lambda:list((out/'clevr_frozen_heads').glob('heads_*'))); s.out_dir=original_out
            tracker.summary(transfer,f'{phase}/clevr_transfer')
            tracker.artifact(out/'clevr_frozen_heads',f'iconqa-{phase}-clevr-transfer')
            run.update(status='complete',method_seconds=timing);write(out/'run.json',run)
            agg=aggregate_run(out,phase)
            tracker.aggregate(agg,phase)
            tracker.figures(out)
            if phase=='discovery':
                frozen=freeze(out,args.output/'iconqa_freeze.json')
            else:
                confirm(out,args.output/'iconqa_freeze.json')
            tracker.artifact(out,f'iconqa-{phase}-results')
            del s
        state.update(status='execution_finished_pending_independent_audit')
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));raise
    finally:
        write(args.output/'campaign_status.json',state)
        if tracker is not None:
            try:
                tracker.run.summary.update({'execution_status':state['status'],'error':state.get('error','')})
                tracker.artifact(args.output,'iconqa-campaign')
            finally:
                tracker.finish('complete' if state['status']=='execution_finished_pending_independent_audit' else 'failed')


if __name__=='__main__':
    main()
