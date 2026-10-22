"""Short train-only gate for full image-group capture and batch-two patching."""
import json
from iconqa_qwen_pilot import write


def group_batch2_guard(session,out):
    import torch
    from vlm_mcqa.core7.hooks import Site,Intervention
    from vlm_mcqa.core7.calibration import TOLERANCE,_kernel_tolerance
    candidates=[v for v in session.variants.values() if v['label_scheme']=='letters'
                and v['variant_kind']=='position' and v['correct_index']==0]
    variant=min(candidates,key=lambda v:v['variant_id'])
    donor=next(v for v in candidates if v['counterfactual_pair_id']==variant['counterfactual_pair_id']
               and v['item_id']!=variant['item_id'])
    prep=session.prepared(variant['variant_id']);dp=session.prepared(donor['variant_id'])
    if len(prep.groups['image'])!=len(dp.groups['image']):raise RuntimeError('image-group geometry mismatch')
    positions=sorted({p for g in ('image','question','options','labels') for p in dp.groups[g]})
    layers=session.layers('group_layer_stride');column={p:i for i,p in enumerate(positions)}
    _,captured=session.adapter.decoder_forward(dp,capture=[Site('resid_post',l) for l in layers],capture_positions=positions)
    donor_cols=[column[p] for p in dp.groups['image']]
    records=[]
    # One real two-row image-block batch on each side of the device boundary.
    for device in (0,1):
        chosen=[l for l in layers if session.adapter.spec.device_map[f'model.language_model.layers.{l}']==device][:2]
        if len(chosen)!=2:raise RuntimeError('insufficient group layers for device guard')
        specs=[Intervention(Site('resid_post',l),prep.groups['image'],captured[f'resid_post:{l}'][0,donor_cols]) for l in chosen]
        reference=torch.cat([session.adapter.decoder_forward(prep,interventions=[i])[0].float().cpu() for i in specs])
        for row,item in enumerate(specs):item.rows=[row]
        for i in range(2):torch.cuda.reset_peak_memory_stats(i)
        actual,_=session.adapter.decoder_forward(prep,batch=2,interventions=specs)
        actual=actual.float().cpu();error=float((reference-actual).abs().max())
        labels=session.label_ids(prep,variant['labels']);same=bool(torch.equal(reference[:,labels].argmax(-1),actual[:,labels].argmax(-1)))
        tolerance=_kernel_tolerance(TOLERANCE['float16'],'float16',reference)
        record={'device':device,'layers':chosen,'max_error':error,'tolerance':tolerance,
                'label_predictions_match':same,'passed':same and error<=tolerance,
                'peak_allocated_gib':[torch.cuda.max_memory_allocated(i)/2**30 for i in range(2)]}
        records.append(record);write(out,{'batch_size':2,'scope':'train-only full group capture and image-block replay','records':records})
        if not record['passed']:raise RuntimeError('batch2 image-group guard failed')
    result={'all_passed':True,'batch_size':2,'records':records,'image_tokens':len(prep.groups['image']),
            'captured_group_layers':layers,'scientific_result':False}
    write(out,result);return result
