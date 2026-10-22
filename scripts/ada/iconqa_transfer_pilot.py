"""Train-only native LLaVA/InternVL FP16 placement, calibration and batch1/2/4."""
import argparse,copy,dataclasses,gc,importlib.metadata,json,os,time
from pathlib import Path
from iconqa_small_inputs import verify_release,digest,SEED,RELEASE_HASH
from iconqa_qwen_pilot import write,align_qwen_rotary_inputs
from iconqa_qwen_small import start_tracker
from stage_iconqa_transfer import MODELS


def placement_for(model_key,layers):
    """Map leaf blocks explicitly; a root fallback makes Accelerate move all weights."""
    boundary=layers//2-1
    placement={'model.vision_tower':0,'model.multi_modal_projector':0,
               'model.language_model.embed_tokens':0,'model.language_model.rotary_emb':0,
               'model.language_model.norm':1,'lm_head':1}
    if model_key.startswith('llava'):placement['model.image_newline']=0
    placement.update({f'model.language_model.layers.{i}':0 if i<boundary else 1 for i in range(layers)})
    return placement,boundary


def configure_llava_sdpa():
    """Turing lacks Flash GQA; repeat KV so efficient SDPA can avoid quadratic scores."""
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from transformers.integrations import sdpa_attention
    gen=torch.Generator(device='cuda:0').manual_seed(SEED)
    q=torch.randn((1,28,64,128),device='cuda:0',dtype=torch.float16,generator=gen)
    k=torch.randn((1,4,64,128),device='cuda:0',dtype=torch.float16,generator=gen)
    v=torch.randn((1,4,64,128),device='cuda:0',dtype=torch.float16,generator=gen)
    with sdpa_kernel(SDPBackend.MATH):
        reference=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=True)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        actual=torch.nn.functional.scaled_dot_product_attention(q,sdpa_attention.repeat_kv(k,7),
                                                               sdpa_attention.repeat_kv(v,7),is_causal=True)
    error=float((reference-actual).abs().max())
    if not torch.allclose(reference,actual,atol=.005,rtol=.01):raise RuntimeError('repeat-KV SDPA check failed')
    sdpa_attention.use_gqa_in_sdpa=lambda attention_mask,key:False
    torch.backends.cuda.enable_math_sdp(False)
    return {'gqa':'explicit repeated KV','math_fallback':False,'efficient_sdpa':True,
            'small_tensor_max_error':error,'native_image_processing':'unchanged'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','dataset','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model',choices=MODELS,required=True);p.add_argument('--execute',action='store_true');args=p.parse_args()
    if not args.execute or args.output.exists():p.error('--execute with fresh output required')
    manifest,release=verify_release(args.dataset)
    receipt=json.loads((args.checkpoint/'verified_checkpoint.json').read_text());model,revision=MODELS[args.model]
    if receipt.get('revision')!=revision or receipt.get('model_id')!=model or receipt.get('status')!='verified':raise ValueError('wrong checkpoint receipt')
    for name,rec in receipt['files'].items():
        path=args.checkpoint/name
        if path.stat().st_size!=rec['bytes'] or (not name.endswith('.safetensors') and digest(path)!=rec['sha256']):raise ValueError('checkpoint changed '+name)
    snapshot=json.loads((args.source_root/'source_snapshot.json').read_text())
    for name,sha in snapshot['files'].items():
        if digest(args.source_root/name)!=sha:raise ValueError('changed source '+name)
    args.output.mkdir(parents=True)
    identity={'model_key':args.model,'model_id':model,'revision':revision,'scientific_result':False,'split':'train_only',
              'status':'starting','dataset_hash':manifest['dataset_hash'],'release':RELEASE_HASH,
              'source_manifest':digest(args.source_root/'source_snapshot.json'),
              'checkpoint_receipt':digest(args.checkpoint/'verified_checkpoint.json'),
              'dtype':'float16','historical_precision':'bfloat16','condition_change':'Ada accessible2080Ti requires separately calibrated FP16',
              'job_id':os.environ.get('SLURM_JOB_ID'),'node':os.environ.get('SLURMD_NODENAME')}
    tracker=None
    try:
        import torch
        from vlm_mcqa.core7.adapters import ModelSpec,VLMAdapter
        from vlm_mcqa.core7.session import Session
        from vlm_mcqa.core7.calibration import run_calibration,TOLERANCE,_kernel_tolerance
        from vlm_mcqa.core7.hooks import Site,Intervention
        if torch.cuda.device_count()!=2:raise RuntimeError('two allocated GPUs required')
        if any(torch.cuda.get_device_capability(i)<(7,5) for i in range(2)):raise RuntimeError('requires Turing-or-newer hardware')
        if args.model.startswith('llava'):identity['sdpa_policy']=configure_llava_sdpa()
        cfg=json.loads((args.checkpoint/'config.json').read_text());text=cfg['text_config'];layers=text['num_hidden_layers']
        placement,n0=placement_for(args.model,layers)
        spec=ModelSpec(key=args.model,family='llava_onevision' if args.model.startswith('llava') else 'internvl',
                       hf_id=str(args.checkpoint),revision=revision,params_b=8.0 if layers==28 else 8.5,
                       lm_backbone='qwen2' if layers==28 else 'qwen3',dtype='float16',device_map=placement)
        identity.update(model_spec=dataclasses.asdict(spec),runtime={n:importlib.metadata.version(n) for n in ('torch','transformers','accelerate')},
                        gpu_names=[torch.cuda.get_device_name(i) for i in range(2)])
        write(args.output/'status.json',identity);tracker=start_tracker(args.output,identity,False,group='iconqa-cf-small-'+args.model,job_type='iconqa-batch-pilot')
        start=time.monotonic();adapter=VLMAdapter.load(spec)
        if any(str(d) not in ('0','1','cuda:0','cuda:1') for d in adapter.model.hf_device_map.values()):raise RuntimeError('CPU/disk offload forbidden')
        parameter_bytes={0:0,1:0}
        for name,param in adapter.model.named_parameters():
            owners=[key for key in placement if name==key or name.startswith(key+'.')]
            if not owners:raise RuntimeError('unmapped model parameter '+name)
            expected=placement[max(owners,key=len)]
            if param.device.type!='cuda' or param.device.index!=expected:
                raise RuntimeError('incorrect parameter placement '+name+' on '+str(param.device))
            parameter_bytes[expected]+=param.numel()*param.element_size()
        identity['verified_parameter_gib']={str(k):v/2**30 for k,v in parameter_bytes.items()}
        if adapter.num_layers!=layers:raise RuntimeError('wrong layer count')
        adapter._iconqa_device_hooks=[adapter.handle.decoder.rotary_emb.register_forward_pre_hook(align_qwen_rotary_inputs,with_kwargs=True)]
        identity.update(load_seconds=time.monotonic()-start,adapter=adapter.describe(),status='calibrating')
        write(args.output/'status.json',identity)
        profile=json.loads((args.source_root/'configs/core7/profiles.json').read_text())['profiles']['full_core7']
        profile.update(batch_size=1,bounded_case_preparation=True)
        class GatedSession(Session):
            def prepared(self,*a,**kw):
                prep=super().prepared(*a,**kw)
                if None in prep.content_ids.values() or len(set(prep.content_ids.values()))!=4:raise RuntimeError('semantic token collision')
                if any(not prep.groups.get(g) for g in ('image','question','options','labels')):raise RuntimeError('unresolved token groups')
                return prep
        s=GatedSession(adapter,args.dataset,args.output,profile,seed=SEED,prepared_cache=2)
        s.items={k:v for k,v in s.items.items() if v['split']=='calibration'}
        s.variants={k:v for k,v in s.variants.items() if v['item_id'] in s.items}
        s.contrasts=[c for c in s.contrasts if c['split']=='calibration'];s.tracker=tracker
        # Bootstrap structural calibration at1; candidate batch checks below are
        # the real batch gates, avoiding an untested fixed batch3 before sizing.
        calibration=run_calibration(s,batch_size=1)
        write(args.output/'calibration_scope.json',{'bootstrap_batch':1,'candidate_batch_checks':'batch_benchmark.json','not_scientific_eligibility':True})
        if not calibration['all_passed']:raise RuntimeError('model structural calibration failed')
        tracker.summary(calibration,'calibration')
        behavior=[]
        for item in sorted(s.items.values(),key=lambda i:i['item_id']):
            v=s.variants_of(item['item_id'],schemes=['letters'])[0];clean=s.clean(v['variant_id'])
            behavior.append({'item_id':item['item_id'],'image_status':item['image_status'],'correct':clean.correct,'margin':clean.readout['margin']})
        write(args.output/'train_behavior.json',behavior)
        identity['status']='benchmarking';write(args.output/'status.json',identity)
        # Same448square inputs/question mean fixed processor tiling geometry;
        # compare native+edited members of one train source in all3alphabets.
        pair=sorted(s.pair_ids('calibration'))[0]
        representatives=[]
        for scheme in ('letters','numbers','rare_letters'):
            for member in ('x','y'):
                representatives.append(next(v for v in s.variants.values() if v['counterfactual_pair_id']==pair
                                       and v['item_id'].endswith('_'+member) and v['label_scheme']==scheme
                                       and v['variant_kind']=='position' and v['correct_index']==0))
        records=[]
        for variant in representatives:
            prep=s.prepared(variant['variant_id'])
            donor=next(v for v in s.variants.values() if v['counterfactual_pair_id']==pair and v['item_id']!=variant['item_id']
                       and v['label_scheme']==variant['label_scheme'] and v['variant_kind']=='position' and v['correct_index']==0)
            clean=s.clean(donor['variant_id']);dp=s.prepared(donor['variant_id'])
            sites=sorted(set((0,n0-1,n0,layers-1)));specs=[]
            for kind in ('resid_post','attn','mlp','head'):
                for layer in sites:
                    specs.append([Intervention(Site(kind,layer),[prep.final_position],clean.sites[f'{kind}:{layer}'],
                                               heads=[0,5,12,24] if kind=='head' else None)])
            specs.append([Intervention(Site('head',l),[prep.final_position],clean.sites[f'head:{l}'],
                                       heads=list(range(adapter.handle.num_heads))) for l in sites])
            if len(prep.groups['image'])!=len(dp.groups['image']):raise RuntimeError('donor visual grid mismatch')
            _,captured=adapter.decoder_forward(dp,capture=[Site('resid_post',n0)],capture_positions=dp.groups['image'])
            specs.append([Intervention(Site('resid_post',n0),prep.groups['image'],captured[f'resid_post:{n0}'][0])])
            def sync():
                for i in range(2):torch.cuda.synchronize(i)
            def execute(batch):
                outputs=[]
                for begin in range(0,len(specs),batch):
                    chunk=specs[begin:begin+batch];interventions=[]
                    for row,items in enumerate(chunk):
                        for item in items:
                            p=copy.copy(item);p.rows=[row];interventions.append(p)
                    logits,_=adapter.decoder_forward(prep,batch=len(chunk),interventions=interventions)
                    outputs.extend(logits.detach().float().cpu().unbind())
                return torch.stack(outputs)
            reference=execute(1);baseline=None;labels=s.label_ids(prep,variant['labels'])
            tolerance=_kernel_tolerance(TOLERANCE['float16'],'float16',reference)
            for batch in (1,2,4):
                r={'batch_size':batch,'scheme':variant['label_scheme'],'image_status':s.items[variant['item_id']]['image_status'],
                   'variant_id':variant['variant_id'],'tokens':prep.num_tokens,'image_tokens':len(prep.groups['image']),
                   'interventions':len(specs),'repeats':2,'tolerance':tolerance}
                try:
                    execute(batch);sync()
                    for i in range(2):torch.cuda.reset_peak_memory_stats(i)
                    start=time.monotonic()
                    for _ in range(2):actual=execute(batch)
                    sync();seconds=time.monotonic()-start
                    if batch==1:baseline=seconds
                    error=float((reference-actual).abs().max());same=bool(torch.equal(reference[:,labels].argmax(-1),actual[:,labels].argmax(-1)))
                    r.update(status='measured',seconds=seconds,speedup=baseline/seconds,max_logit_error=error,
                             label_predictions_match=same,equivalence_passed=same and error<=tolerance,
                             peak_allocated_gib=[torch.cuda.max_memory_allocated(i)/2**30 for i in range(2)],
                             peak_reserved_gib=[torch.cuda.max_memory_reserved(i)/2**30 for i in range(2)])
                except torch.OutOfMemoryError:
                    r.update(status='oom',equivalence_passed=False);gc.collect();torch.cuda.empty_cache()
                if batch==1 and r['status']=='oom':raise RuntimeError('batch1 benchmark exceeds GPU memory')
                records.append(r);write(args.output/'batch_benchmark.json',{'scientific_result':False,'records':records})
                tracker.log(r,r['scheme']+'/'+r['image_status']);print('[transfer pilot]',r,flush=True)
            del captured,specs,reference
        valid=[b for b in (1,2,4) if len([r for r in records if r['batch_size']==b])==6
               and all(r.get('equivalence_passed') for r in records if r['batch_size']==b)]
        speed={b:sum(r['speedup'] for r in records if r['batch_size']==b)/6 for b in valid}
        recommended=max(valid,key=lambda b:speed[b]) if valid else None
        summary={'status':'completed','recommended_batch':recommended,'mean_speedup':speed,'records':len(records),
                 'scope':'train-only representative mixed interventions; full campaign and32pair eligibility untested'}
        write(args.output/'pilot_summary.json',summary);tracker.summary(summary,'pilot');tracker.artifact(args.output,'iconqa-'+args.model+'-batch-pilot')
        identity.update(status='complete',recommended_batch=recommended)
    except BaseException as exc:identity.update(status='failed',error=repr(exc));raise
    finally:
        write(args.output/'status.json',identity)
        if tracker:tracker.finish('complete' if identity['status']=='complete' else 'failed')


if __name__=='__main__':main()
