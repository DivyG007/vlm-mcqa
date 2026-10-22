"""Small train-only mixed-intervention throughput/equivalence benchmark."""
import copy
import time
import torch
from iconqa_qwen_pilot import write
from vlm_mcqa.core7.hooks import Site, Intervention
from vlm_mcqa.core7.calibration import TOLERANCE, _kernel_tolerance
from vlm_mcqa.core7.tracking import Tracker


def benchmark(session,out,identity,*,tracker=None,batches=(1,2,4)):
    own_tracker=tracker is None
    tracker=tracker or Tracker.start(enabled=True,out_dir=out,name=out.name,job_type='batch-benchmark',
                          config=identity,tags=['iconqa','train-only','batch-benchmark'])
    records=[]
    def sync():
        for device in range(2):
            torch.cuda.synchronize(device)
    try:
        # Longest prompt in each alphabet; no validation/test access or science selection.
        selected=[]
        for scheme in ('letters','numbers','rare_letters'):
            candidates=[v for v in session.variants.values() if v['label_scheme']==scheme]
            selected.append(max(candidates,key=lambda v:session.prepared(v['variant_id']).num_tokens))
        for variant in selected:
            prep=session.prepared(variant['variant_id'])
            donor=next(v for v in session.variants.values() if v['item_id']!=variant['item_id']
                       and v['label_scheme']==variant['label_scheme'])
            clean=session.clean(donor['variant_id'])
            specs=[]
            for kind in ('resid_post','attn','mlp','head'):
                for layer in (0,12,13,27):
                    site=Site(kind,layer)
                    specs.append([Intervention(site,[prep.final_position],clean.sites[site.key],
                                               heads=[0,5,12,24] if kind=='head' else None)])
            # Full visual-block transplantation, the largest token-group intervention.
            dp=session.prepared(donor['variant_id'])
            _,captured=session.adapter.decoder_forward(dp,capture=[Site('resid_post',13)],
                                                       capture_positions=dp.groups['image'])
            specs.append([Intervention(Site('resid_post',13),prep.groups['image'],captured['resid_post:13'][0])])
            def execute(batch):
                outputs=[]
                for start in range(0,len(specs),batch):
                    chunk=specs[start:start+batch]; interventions=[]
                    for row,items in enumerate(chunk):
                        for item in items:
                            placed=copy.copy(item); placed.rows=[row]; interventions.append(placed)
                    logits,_=session.adapter.decoder_forward(prep,batch=len(chunk),interventions=interventions)
                    outputs.extend(logits.detach().float().cpu().unbind())
                return torch.stack(outputs)
            reference=execute(1)
            tolerance=_kernel_tolerance(TOLERANCE[session.adapter.spec.dtype],session.adapter.spec.dtype,reference)
            baseline_seconds=None
            for batch in batches:
                record={'scheme':variant['label_scheme'],'variant_id':variant['variant_id'],
                        'donor_variant_id':donor['variant_id'],'tokens':prep.num_tokens,'batch_size':batch,
                        'interventions':len(specs),'repeats':3,'tolerance':tolerance}
                try:
                    execute(batch); sync()  # untimed warmup
                    for device in range(2): torch.cuda.reset_peak_memory_stats(device)
                    start=time.monotonic()
                    for _ in range(3): actual=execute(batch)
                    sync(); seconds=time.monotonic()-start
                    error=float((actual-reference).abs().max())
                    labels=list(prep.label_ids.values()) if isinstance(prep.label_ids,dict) else list(prep.label_ids)
                    same=bool(torch.equal(actual[:,labels].argmax(-1),reference[:,labels].argmax(-1)))
                    if batch==1: baseline_seconds=seconds
                    record.update(status='measured',seconds=seconds,speedup=baseline_seconds/seconds,
                                  max_logit_error=error,label_predictions_match=same,
                                  equivalence_passed=error<=tolerance and same,
                                  peak_allocated_gib=[torch.cuda.max_memory_allocated(i)/2**30 for i in range(2)],
                                  peak_reserved_gib=[torch.cuda.max_memory_reserved(i)/2**30 for i in range(2)])
                except torch.OutOfMemoryError:
                    record.update(status='oom',equivalence_passed=False)
                    torch.cuda.empty_cache()
                records.append(record)
                write(out/'batch_benchmark.json',{'scientific_result':False,'scope':'train-only representative mixed interventions; not full campaign certification',
                                                 'records':records})
                tracker.log(record,f"{variant['label_scheme']}/batch{batch}")
                print('[batch benchmark]',record,flush=True)
            del captured, specs, reference
        tracker.artifact(out,'iconqa-batch-benchmark')
    finally:
        if own_tracker:tracker.finish('complete' if len(records)==3*len(batches) else 'failed')
