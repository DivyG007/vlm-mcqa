"""Qwen small controlled IconQA campaign. Explicit --execute, fresh outputs/resume."""
import argparse,dataclasses,hashlib,importlib.metadata,json,os,time,uuid
from pathlib import Path
from collections import defaultdict
from iconqa_qwen_pilot import write,align_qwen_rotary_inputs
from iconqa_small_inputs import verify_release,select_pairs,digest,SEED,RELEASE_HASH
from iconqa_checkpoints import Blocks

REVISION='cc594898137f460bfe9f0759e9844b3ce807cfb5'
CLEVR={'head_layers':[19,24,26,27],'head_candidates':[[26,24],[24,27],[24,26],[24,21]],'head_set_sizes':[4]}


def verify_checkpoint(root):
    receipt=json.loads((root/'verified_checkpoint.json').read_text())
    if receipt.get('revision')!=REVISION or receipt.get('status')!='verified':raise ValueError('pinned verified Qwen required')
    for name,r in receipt['files'].items():
        path=root/name
        if path.stat().st_size!=r['bytes'] or (not name.endswith('.safetensors') and digest(path)!=r['sha256']):
            raise ValueError('checkpoint changed: '+name)


class LocalFirstTracker:
    """Preserve W&B records without turning publication errors into science failures."""
    def __init__(self,tracker,out):self.tracker=tracker;self.out=out
    @property
    def url(self):return self.tracker.url
    def __getattr__(self,key):
        def call(*args,**kwargs):
            try:return getattr(self.tracker,key)(*args,**kwargs)
            except Exception as exc:
                with (self.out/'tracking_errors.jsonl').open('a') as f:
                    f.write(json.dumps({'operation':key,'error':repr(exc),'unix':time.time()})+'\n')
                print('[tracking] local outputs retained; publication operation failed: '+key,flush=True)
        return call


def start_tracker(out,identity,resume,*,group='iconqa-cf-small-qwen',job_type='iconqa-small-core7'):
    from vlm_mcqa.core7.tracking import Tracker,destination
    os.environ.setdefault('CORE7_WANDB_ENTITY','models-iiit-hyderabad5465')
    os.environ.setdefault('CORE7_WANDB_PROJECT','anlp-vlm-mcqa-staging')
    entity,project=destination();p=out/'wandb_identity.json'
    saved=json.loads(p.read_text()) if p.exists() else {'id':uuid.uuid4().hex[:12],'entity':entity,'project':project}
    if (saved['entity'],saved['project'])!=(entity,project):raise ValueError('tracking destination changed on resume')
    write(p,saved)
    options=dict(enabled=True,out_dir=out,name=out.name,job_type=job_type,config=identity,
                 run_id=saved['id'],group=group,tags=['iconqa','controlled-appearance','reduced-core7'])
    mode=os.environ.get('WANDB_MODE','online')
    if mode not in ('online','offline'):raise ValueError('W&B tracking must be online or offline')
    try:t=Tracker.start(**options,mode=mode,resume='allow' if resume else 'never',init_timeout=20)
    except Exception as exc:
        write(out/'wandb_online_failure.json',{'error':repr(exc),'offline_fallback':True})
        import wandb
        try:wandb.finish(exit_code=1,quiet=True)
        except Exception:pass
        mode='offline';t=Tracker.start(**options,mode=mode,init_timeout=20)
    write(out/'tracking_state.json',{'mode':mode,'id':saved['id'],'url':t.url,'pending_sync':mode=='offline'})
    return LocalFirstTracker(t,out)


def edit_control_panel(s,phase,selected,clevr=None):
    """Same-answer behavior plus fixed-text donor patches at pre-frozen CLEVR heads."""
    from vlm_mcqa.core7.hooks import Site,Intervention
    clevr=CLEVR if clevr is None else clevr
    controls={c['pair_id']:c for c in json.loads((s.data_dir/'same_answer_controls.json').read_text())}
    rows=[]
    for index,pair in enumerate(selected):
        native=next(v for v in s.variants.values() if v['counterfactual_pair_id']==pair
                    and v['item_id'].endswith('_x') and v['label_scheme']=='letters'
                    and v['variant_kind']=='position' and v['correct_index']==index%4)
        edited_item=next(i for i in s.items.values() if i['counterfactual_pair_id']==pair and i['pair_member']=='y')
        prep=s.prepared(native['variant_id']);ids=s.label_ids(prep,native['labels'])
        target=native['option_contents'].index(edited_item['semantic_answer']);recipient=native['correct_index']
        native_logits,_=s.adapter.decoder_forward(prep)
        def diff(logits):return float(logits[ids[target]]-logits[ids[recipient]])
        baseline=diff(native_logits[0]);conditions={}
        for condition,image in [('counterfactual',edited_item['image']),('same_answer',controls[pair]['image'])]:
            # Same displayed text/order for both native and donor full forwards.
            donor=s.adapter.prepare(native,s.image(image))
            if len(donor.groups.get('image',[]))!=len(prep.groups['image']) or donor.num_tokens!=prep.num_tokens:
                raise RuntimeError('fixed-text control geometry changed')
            logits,captured=s.adapter.decoder_forward(donor,capture=[Site('head',l) for l in clevr['head_layers']],
                                                      capture_positions=[donor.final_position])
            interventions=[];by_layer=defaultdict(list)
            for l,h in clevr['head_candidates']:by_layer[l].append(h)
            for l,hs in by_layer.items():
                interventions.append(Intervention(Site('head',l),[prep.final_position],captured[f'head:{l}'][0,0],heads=hs))
            patched,_=s.adapter.decoder_forward(prep,interventions=interventions)
            conditions[condition]={'clean_diff':diff(logits[0]),'patched_diff':diff(patched[0]),
                                   'patched_minus_native':diff(patched[0])-baseline,
                                   'forced_native_answer_correct':int(logits[0,ids].argmax())==recipient}
        rows.append({'pair_id':pair,'native_variant_id':native['variant_id'],'native_diff':baseline,
                     'frozen_heads':clevr['head_candidates'],'conditions':conditions})
    (s.out_dir/f'edit_controls_{phase}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    summary={'source_pairs':len(rows),'same_answer_accuracy':sum(r['conditions']['same_answer']['forced_native_answer_correct'] for r in rows)/len(rows),
             'counterfactual_fixed_text_changes_answer_rate':sum(not r['conditions']['counterfactual']['forced_native_answer_correct'] for r in rows)/len(rows),
             'same_answer_mean_patch_change':sum(r['conditions']['same_answer']['patched_minus_native'] for r in rows)/len(rows),
             'counterfactual_mean_patch_change':sum(r['conditions']['counterfactual']['patched_minus_native'] for r in rows)/len(rows)}
    write(s.out_dir/f'edit_controls_{phase}_summary.json',summary);return summary


def behavior_with_breakdown(s,phase,run_behavior):
    result=run_behavior(s,split=phase,n_items=128,robust_items=128,compliance_items=128,controls=True)
    groups=defaultdict(list)
    for line in (s.out_dir/f'behavior_{phase}.jsonl').open():
        r=json.loads(line);item=s.items[r['item_id']]
        key='/'.join((item['image_status'],item['semantic_answer'],r['scheme'],r['condition']))
        groups[key].append(bool(r['correct']))
    breakdown={k:{'accuracy':sum(v)/len(v),'variant_records':len(v)} for k,v in groups.items()}
    write(s.out_dir/f'behavior_{phase}_native_edited.json',breakdown)
    result['native_edited_class_breakdown']=breakdown;return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','dataset','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--preflight',action='store_true');p.add_argument('--calibration-only',action='store_true')
    p.add_argument('--batch-size',type=int,choices=[4],default=4);args=p.parse_args()
    manifest,release=verify_release(args.dataset);verify_checkpoint(args.checkpoint)
    snapshot=json.loads((args.source_root/'source_snapshot.json').read_text())
    for name,expected in snapshot['files'].items():
        if digest(args.source_root/name)!=expected:raise ValueError('source snapshot changed: '+name)
    if args.preflight:
        print(json.dumps({'status':'preflight_passed_no_inference','dataset_hash':manifest['dataset_hash'],
                          'release':RELEASE_HASH,'source_commit':snapshot['source_git_commit'],'batch_size':4,'gpus':2}));return
    if not args.execute or args.output.exists()!=args.resume:p.error('--execute and fresh output or --resume required')
    identity={'source_manifest':digest(args.source_root/'source_snapshot.json'),'release':RELEASE_HASH,
              'checkpoint_receipt':digest(args.checkpoint/'verified_checkpoint.json'),'batch_size':4,'seed':SEED,
              'protocol':'iconqa_cf_core7_small_v1','calibration_only':args.calibration_only,
              'runtime':{n:importlib.metadata.version(n) for n in ('torch','transformers','accelerate')}}
    args.output.mkdir(parents=True,exist_ok=args.resume)
    binding=args.output/'resume_identity.json'
    if args.resume and json.loads(binding.read_text())!=identity:raise ValueError('resume identity changed')
    write(binding,identity);state={'status':'starting','scientific_audit':'pending',**identity};tracker=None
    def stage(name):
        state.update(status='running',stage=name,updated_unix=time.time());write(args.output/'campaign_status.json',state)
        print('[small] '+name,flush=True)
    try:
        tracker=start_tracker(args.output,identity,args.resume);state['wandb_url']=tracker.url
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
        placement={'model.visual':0,'model.language_model.embed_tokens':0,'model.language_model.rotary_emb':0,
                   'model.language_model.norm':1,'lm_head':1}
        placement.update({f'model.language_model.layers.{i}':0 if i<=12 else 1 for i in range(28)})
        spec=ModelSpec(key='qwen25vl_7b',family='qwen_vl',hf_id=str(args.checkpoint),revision=REVISION,params_b=8.3,
                       lm_backbone='qwen2.5',dtype='float16',device_map=placement,
                       processor_kwargs={'min_pixels':448**2,'max_pixels':448**2})
        stage('model_loading');adapter=VLMAdapter.load(spec)
        adapter._iconqa_device_hooks=[adapter.handle.decoder.rotary_emb.register_forward_pre_hook(align_qwen_rotary_inputs,with_kwargs=True)]
        profile=json.loads((args.source_root/'configs/core7/profiles.json').read_text())['profiles']['full_core7']
        profile.update(pairs=32,behavior_items=128,batch_size=4,bounded_case_preparation=True,
                       head_screen_contrasts=64,ablation_recipients=32,topk=[1,2,4,8,16])
        write(args.output/'run.json',provenance({'model_key':'qwen25vl_7b','model_spec':dataclasses.asdict(spec),
              'adapter':adapter.describe(),'profile':profile,'profile_name':'iconqa_cf_core7_small_v1',
              'dataset_hash':manifest['dataset_hash'],'image_aware_release_hash':RELEASE_HASH,
              'job_id':os.environ.get('SLURM_JOB_ID'),'status':'running'}))
        class GatedSession(Session):
            def prepared(self,*a,**kw):
                prep=super().prepared(*a,**kw)
                if None in prep.content_ids.values() or len(set(prep.content_ids.values()))!=4 or len(prep.groups.get('image',[]))!=256:
                    raise RuntimeError('semantic/image-token gate')
                if any(not prep.groups.get(k) for k in ('question','options','labels')):raise RuntimeError('token span gate')
                return prep
        reduced=json.loads((args.dataset/'reduced_contrasts.json').read_text())
        def session(phase):
            s=GatedSession(adapter,args.dataset,args.output/phase,profile,seed=SEED,prepared_cache=2)
            s.items={k:v for k,v in s.items.items() if v['split']==phase}
            s.variants={k:v for k,v in s.variants.items() if v['item_id'] in s.items}
            s.contrasts=[c for c in reduced if c['split']==phase];s.tracker=tracker;return s
        stage('train_calibration');train=session('calibration');calibration=run_calibration(train)
        tracker.summary(calibration,'calibration')
        if not calibration['all_passed']:raise RuntimeError('real calibration failed')
        from iconqa_batch_benchmark import benchmark
        stage('train_batch4_equivalence');benchmark(train,train.out_dir,identity,tracker=tracker,batches=(1,4))
        evidence=json.loads((train.out_dir/'batch_benchmark.json').read_text())
        records=[r for r in evidence['records'] if r['batch_size']==4]
        if len(records)!=3 or any(not r.get('equivalence_passed') for r in records):raise RuntimeError('new-release batch4 gate failed')
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
            run=provenance({'model_key':'qwen25vl_7b','model_spec':dataclasses.asdict(spec),'num_layers':adapter.num_layers,
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
                    write(s.out_dir/'source_freeze.json',{'freeze_hash':'4f991401f5cb7239',**CLEVR})
                    return heads(CLEVR)
                finally:s.out_dir=original
            actions=[('behavior',lambda:behavior_with_breakdown(s,phase,run_behavior),('behavior_','compliance_')),
                     ('lens',lambda:run_lens(s,split=phase,n_items=128),('lens_','component_projection_')),
                     ('patching',lambda:run_patching(s,split=phase),('patching_',)),
                     ('components',lambda:run_components(s,split=phase),('components_','component_ablation_')),
                     ('heads',lambda:heads(frozen),('heads_',)),
                     ('clevr_transfer',clevr_transfer,None),
                     ('edit_controls',lambda:edit_control_panel(s,phase,coverage['pair_ids']),('edit_controls_',))]
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
            tracker.artifact(out,'iconqa-small-'+phase)
            tracker.artifact(out/'clevr_frozen_heads','iconqa-small-'+phase+'-clevr')
            del s
        state['status']='execution_finished_pending_independent_audit'
    except BaseException as exc:state.update(status='failed',error=repr(exc));raise
    finally:
        write(args.output/'campaign_status.json',state)
        if tracker:
            tracker.artifact(args.output,'iconqa-small-campaign')
            tracker.finish('complete' if state['status'] in ('execution_finished_pending_independent_audit','calibration_passed_no_science') else 'failed')


if __name__=='__main__':main()
