"""Reduced Core-7 LLaVA-OneVision-7B-SI campaign; explicit execute, hash-bound resume."""
import argparse,dataclasses,importlib.metadata,json,os,time
from pathlib import Path
from iconqa_qwen_pilot import write,align_qwen_rotary_inputs
from iconqa_small_inputs import verify_release,select_pairs,digest,SEED,RELEASE_HASH
from iconqa_checkpoints import Blocks
from iconqa_qwen_small import start_tracker,edit_control_panel,behavior_with_breakdown
from iconqa_llava_contract import verify_checkpoint,verify_pilot,verify_loaded_placement
from iconqa_transfer_pilot import placement_for,configure_llava_sdpa
from iconqa_llava_memory_guard import group_batch2_guard
REVISION='cae7503dce88887557f73db1882328d21d9a7e21'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','dataset','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--pilot-evidence',type=Path,required=True);p.add_argument('--execute',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--preflight',action='store_true');p.add_argument('--calibration-only',action='store_true')
    p.add_argument('--batch-size',type=int,choices=[2],default=2);args=p.parse_args()
    manifest,release=verify_release(args.dataset);verify_checkpoint(args.checkpoint);pilot=verify_pilot(args.pilot_evidence)
    snapshot=json.loads((args.source_root/'source_snapshot.json').read_text())
    for name,expected in snapshot['files'].items():
        if digest(args.source_root/name)!=expected:raise ValueError('source snapshot changed: '+name)
    if args.preflight:
        print(json.dumps({'status':'preflight_passed_no_inference','dataset_hash':manifest['dataset_hash'],
                          'release':RELEASE_HASH,'source_commit':snapshot['source_git_commit'],'batch_size':2,'gpus':2,'pilot_evidence':pilot['sha256']}));return
    if not args.execute or args.output.exists()!=args.resume:p.error('--execute and fresh output or --resume required')
    identity={'source_manifest':digest(args.source_root/'source_snapshot.json'),'release':RELEASE_HASH,
              'checkpoint_receipt':digest(args.checkpoint/'verified_checkpoint.json'),'batch_size':2,'seed':SEED,
              'protocol':'iconqa_cf_core7_small_v1','model_key':'llava_ov_7b_si','attention_policy':pilot['sdpa_policy'],'pilot_evidence':pilot['sha256'],'clevr_freeze':digest(args.source_root/'configs/core7/iconqa_small_llava_clevr_freeze.json'),'calibration_only':args.calibration_only,
              'runtime':{n:importlib.metadata.version(n) for n in ('torch','transformers','accelerate')}}
    if identity['runtime']!=pilot['runtime']:raise ValueError('runtime changed since admitted pilot')
    args.output.mkdir(parents=True,exist_ok=args.resume)
    binding=args.output/'resume_identity.json'
    if args.resume and json.loads(binding.read_text())!=identity:raise ValueError('resume identity changed')
    write(binding,identity);state={'status':'starting','scientific_audit':'pending',**identity};tracker=None
    def stage(name):
        state.update(status='running',stage=name,updated_unix=time.time());write(args.output/'campaign_status.json',state)
        print('[small] '+name,flush=True)
    try:
        tracker=start_tracker(args.output,identity,args.resume,group='iconqa-cf-small-llava_ov_7b_si');state['wandb_url']=tracker.url
        import torch
        from vlm_mcqa.core7.adapters import ModelSpec,VLMAdapter
        from vlm_mcqa.core7.session import Session,provenance
        from vlm_mcqa.core7.calibration import run_calibration
        from vlm_mcqa.core7.behavior import run_behavior
        from vlm_mcqa.core7.lens import run_lens
        from vlm_mcqa.core7.patching import run_patching
        from vlm_mcqa.core7.components import run_components
        from vlm_mcqa.core7.heads import run_heads
        from vlm_mcqa.core7.aggregate import aggregate_run,freeze,confirm
        if torch.cuda.device_count()!=2 or any(torch.cuda.get_device_capability(i)<(7,5) for i in range(2)):
            raise RuntimeError('requires two Turing-or-newer GPUs')
        state['sdpa_policy']=configure_llava_sdpa()
        placement,_=placement_for('llava_ov_7b_si',28)
        spec=ModelSpec(key='llava_ov_7b_si',family='llava_onevision',hf_id=str(args.checkpoint),revision=REVISION,
                       params_b=8.0,lm_backbone='qwen2',dtype='float16',device_map=placement)
        clevr=json.loads((args.source_root/'configs/core7/iconqa_small_llava_clevr_freeze.json').read_text())
        if clevr['freeze_hash']!='e7b66bcb947ff29c' or clevr['model_key']!='llava_ov_7b_si':raise ValueError('wrong historical freeze')
        stage('model_loading');adapter=VLMAdapter.load(spec)
        verify_loaded_placement(adapter,placement)
        if adapter.num_layers!=28 or adapter.handle.num_heads!=28:raise RuntimeError('wrong LLaVA architecture')
        adapter._iconqa_device_hooks=[adapter.handle.decoder.rotary_emb.register_forward_pre_hook(align_qwen_rotary_inputs,with_kwargs=True)]
        profile=json.loads((args.source_root/'configs/core7/profiles.json').read_text())['profiles']['full_core7']
        profile.update(pairs=32,behavior_items=128,batch_size=2,bounded_case_preparation=True,
                       head_screen_contrasts=64,ablation_recipients=32,topk=[1,2,4,8,16])
        write(args.output/'run.json',provenance({'model_key':'llava_ov_7b_si','model_spec':dataclasses.asdict(spec),
              'adapter':adapter.describe(),'profile':profile,'profile_name':'iconqa_cf_core7_small_v1',
              'dataset_hash':manifest['dataset_hash'],'image_aware_release_hash':RELEASE_HASH,
              'job_id':os.environ.get('SLURM_JOB_ID'),'precision_condition':'FP16; historical CLEVR BF16','pilot_evidence':pilot,'status':'running'}))
        class GatedSession(Session):
            def prepared(self,*a,**kw):
                prep=super().prepared(*a,**kw)
                if None in prep.content_ids.values() or len(set(prep.content_ids.values()))!=4 or len(prep.groups.get('image',[]))!=pilot['image_tokens']:
                    raise RuntimeError('semantic/image-token gate')
                if prep.num_tokens>pilot['max_tokens']:raise RuntimeError('prompt exceeds pilot memory envelope')
                if any(not prep.groups.get(k) for k in ('question','options','labels')):raise RuntimeError('token span gate')
                return prep
        reduced=json.loads((args.dataset/'reduced_contrasts.json').read_text())
        def session(phase):
            s=GatedSession(adapter,args.dataset,args.output/phase,profile,seed=SEED,prepared_cache=2)
            s.items={k:v for k,v in s.items.items() if v['split']==phase}
            s.variants={k:v for k,v in s.variants.items() if v['item_id'] in s.items}
            s.contrasts=[c for c in reduced if c['split']==phase];s.tracker=tracker;return s
        stage('train_calibration');train=session('calibration');calibration=run_calibration(train,batch_size=2)
        tracker.summary(calibration,'calibration')
        if not calibration['all_passed']:raise RuntimeError('real calibration failed')
        stage('train_batch2_image_group_guard')
        guard=group_batch2_guard(train, args.output/'batch2_memory_guard.json')
        tracker.summary(guard,'batch2_memory_guard')
        # Reuse the completed exact-model/release/precision batch pilot instead of repeating it.
        write(args.output/'pilot_evidence.json',pilot)
        del train
        if args.calibration_only:state['status']='calibration_passed_no_science';return
        frozen=None
        for phase in ('discovery','confirmation'):
            stage(phase+'/eligibility');s=session(phase);out=s.out_dir
            cs,coverage,head_cases,recipients=select_pairs(s,phase,reduced)
            write(out/'eligibility.json',coverage)
            if coverage['pairs_used']!=32:raise RuntimeError(f'{phase}: insufficient32pair eligibility')
            s.contrasts=cs;s.profile['ablation_recipient_ids']=recipients
            write(out/'selected_contrasts.json',cs);write(out/'head_cases.json',head_cases)
            write(out/'ablation_recipients.json',recipients);write(out/'calibration.json',calibration)
            run=provenance({'model_key':'llava_ov_7b_si','model_spec':dataclasses.asdict(spec),'num_layers':adapter.num_layers,
                            'dataset_hash':manifest['dataset_hash'],'image_aware_release_hash':RELEASE_HASH,
                            'profile':s.profile,'profile_name':'iconqa_cf_core7_small_v1','split':phase,'status':'running',
                            'wandb_url':tracker.url,'methods':['behavior','lens','patching','components','heads']})
            write(out/'run.json',run)
            blocks=Blocks(out/'.checkpoints',{**identity,'phase':phase,'selection':digest(out/'selected_contrasts.json'),
                          'head_cases':digest(out/'head_cases.json'),'ablation':digest(out/'ablation_recipients.json')},s.rng)
            def heads(freeze_value):
                original=s.contrasts;s.contrasts=head_cases
                try:return run_heads(s,split=phase,freeze=freeze_value)
                finally:s.contrasts=original
            def clevr_transfer():
                original=s.out_dir;s.out_dir=out/'clevr_frozen_heads';s.out_dir.mkdir(exist_ok=True)
                try:
                    write(s.out_dir/'source_freeze.json',clevr)
                    return heads(clevr)
                finally:s.out_dir=original
            actions=[('behavior',lambda:behavior_with_breakdown(s,phase,run_behavior),('behavior_','compliance_')),
                     ('lens',lambda:run_lens(s,split=phase,n_items=128),('lens_','component_projection_')),
                     ('patching',lambda:run_patching(s,split=phase),('patching_',)),
                     ('components',lambda:run_components(s,split=phase),('components_','component_ablation_')),
                     ('heads',lambda:heads(frozen),('heads_',)),
                     ('clevr_transfer',clevr_transfer,None),
                     ('edit_controls',lambda:edit_control_panel(s,phase,coverage['pair_ids'],clevr),('edit_controls_',))]
            timings={}
            for name,action,prefixes in actions:
                stage(phase+'/'+name)
                def artifacts():
                    if prefixes is None:return list((out/'clevr_frozen_heads').glob('*.json*'))
                    return [p for p in out.iterdir() if p.is_file() and p.name.startswith(prefixes) and not p.name.endswith('_completion.json')]
                reused=(out/'.checkpoints'/(name+'.json')).exists();start=time.monotonic()
                result=blocks.run(name,action,artifacts);seconds=time.monotonic()-start
                completion=out/(name+'_completion.json')
                if not reused:write(completion,{'seconds':seconds,'status':'complete','result':result})
                timings[name]=json.loads(completion.read_text())['seconds']
                tracker.log(result,phase+'/'+name);tracker.summary(result,phase+'/'+name)
                tracker.log({phase+'/'+name+'/invocation_seconds':seconds,phase+'/'+name+'/reused_checkpoint':reused})
                tracker.log({f'gpu/{i}/peak_allocated_gib':torch.cuda.max_memory_allocated(i)/2**30 for i in range(2)})
                tracker.log({f'gpu/{i}/peak_reserved_gib':torch.cuda.max_memory_reserved(i)/2**30 for i in range(2)})
                if name=='behavior' and not (result['worst_position_accuracy']>.90 and result['clean_minus_shuffled']>=.30 and result['worst_scheme_position_accuracy']>=.45):
                    raise RuntimeError('behavior eligibility failed')
                if name=='patching' and (result['m1_rows']!=16128 or result['group_patching']['rows']!=15360 or result['group_patching']['skipped_size_mismatch']):
                    raise RuntimeError('residual/group coverage mismatch')
                if name=='components' and (result['patch_rows']!=21504 or result['ablation_rows']!=1792):raise RuntimeError('component work counts mismatch')
                if name in ('heads','clevr_transfer') and result['cases']['pairs_used']!=32:raise RuntimeError('head source coverage mismatch')
            run.update(status='complete',method_seconds=timings);write(out/'run.json',run)
            agg=aggregate_run(out,phase);tracker.aggregate(agg,phase)
            if phase=='discovery':frozen=freeze(out,args.output/'iconqa_freeze.json')
            else:confirm(out,args.output/'iconqa_freeze.json')
            tracker.artifact(out,'iconqa-small-llava-'+phase)
            tracker.artifact(out/'clevr_frozen_heads','iconqa-small-llava-'+phase+'-clevr')
            del s
        state['status']='execution_finished_pending_independent_audit'
    except BaseException as exc:state.update(status='failed',error=repr(exc));raise
    finally:
        write(args.output/'campaign_status.json',state)
        if tracker:
            tracker.artifact(args.output,'iconqa-small-campaign')
            tracker.finish('complete' if state['status'] in ('execution_finished_pending_independent_audit','calibration_passed_no_science') else 'failed')


if __name__=='__main__':main()
