"""Independent on-disk counterfactual audit and explicit visual-review finalizer."""
import argparse,json,hashlib
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from PIL import Image,ImageDraw
from probe_iconqa_marbles import objects,answer
from build_iconqa_cf_small import sha,dump,pixels,letterbox,SEED
from vlm_mcqa.clevr_mcq4.variants import validate
from vlm_mcqa.clevr_mcq4.build import expand


def audit(root,finalize=False):
    read=lambda name:[json.loads(s) for s in (root/name).read_text().splitlines()]
    items=read('items.jsonl');vs=read('variants.jsonl');vmap={v['variant_id']:v for v in vs}
    edits=json.loads((root/'edits.json').read_text());controls=json.loads((root/'same_answer_controls.json').read_text())
    reduced=json.loads((root/'reduced_contrasts.json').read_text())
    structural=validate(items,vs,reduced)
    seen={};pairset=set();min_iou=1.;by_pair=defaultdict(list)
    for c in reduced:by_pair[c['pair_id']].append(c)
    expected={'position':4,'content':4,'content_and_position':4,'identity':4,'random_pair':2}
    for r in edits:
        pair=r['pair_id'];pairset.add(pair)
        x=Image.open(root/'native'/f'{pair}_x.png').convert('RGB');y=Image.open(root/'native'/f'{pair}_y.png').convert('RGB')
        a,b=np.array(x),np.array(y);delta=np.any(a!=b,2);mask=np.array(Image.open(root/'masks'/f'{pair}.png'))>0
        if not np.array_equal(delta,mask):raise ValueError('mask mismatch '+pair)
        left,top,right,bottom=r['edit_bbox'];outside=delta.copy();outside[top:bottom,left:right]=False
        if outside.any() or not delta.any():raise ValueError('locality '+pair)
        if sha(r['source_path'])!=r['source_sha256']:raise ValueError('source changed '+pair)
        source=Image.open(r['source_path']).convert('RGB')
        if not np.array_equal(a,np.array(source)):raise ValueError('factual is not native '+pair)
        ox,oy=objects(x),objects(y)
        if len(ox)!=len(oy) or len(ox)!=r['total']:raise ValueError('count changed')
        black=[sum(o['color']=='black' for o in obs) for obs in (ox,oy)]
        if abs(black[0]-black[1])!=1:raise ValueError('not one marble')
        if [answer(v,len(ox)) for v in black]!=[r['native_answer'],r['edited_answer']]:raise ValueError('wrong truth')
        if r['native_answer']==r['edited_answer']:raise ValueError('answer did not change')
        for im,member in ((x,'x'),(y,'y')):
            key=pixels(im)
            if key in seen:raise ValueError('duplicate image component '+pair)
            seen[key]=pair
            p=root/'images'/f'{pair}_{member}.png'
            if not np.array_equal(np.array(letterbox(im)[0]),np.array(Image.open(p))):raise ValueError('preprocessing mismatch')
        min_iou=min(min_iou,r['shape_iou'])
        counts=Counter(c['contrast_type'] for c in by_pair[pair])
        if dict(counts)!=expected:raise ValueError('reduced allocation '+pair)
        for kind in ('position','content','content_and_position','identity'):
            edges={(c['donor_variant_id'],c['recipient_variant_id']) for c in by_pair[pair] if c['contrast_type']==kind}
            if any((b,a) not in edges for a,b in edges):raise ValueError('missing reverse direction')
            if kind in ('position','content_and_position'):
                positions=Counter(vmap[a]['correct_index'] for a,b in edges)
                if positions!={0:1,1:1,2:1,3:1}:raise ValueError('unbalanced per-pair positions')
    for c in reduced:
        d,r=vmap[c['donor_variant_id']],vmap[c['recipient_variant_id']]
        if d['split']!=r['split']:raise ValueError('contrast crosses split')
        if c['contrast_type']=='content_and_position' and r['option_contents'][c['content_index']]!=d['semantic_answer']:
            raise ValueError('factorial content mapping')
    for c in controls:
        im=Image.open(root/'controls'/f"{c['pair_id']}.png").convert('RGB')
        # Detect counts before resize from saved checked native-object metadata;
        # independently check processed control truth as an additional gate.
        obs=objects(im);black=sum(o['color']=='black' for o in obs)
        if answer(black,len(obs))!=c['answer']:raise ValueError('control truth '+c['pair_id'])
    report={'status':'passed','pairs':len(edits),'items':len(items),'reduced_contrasts':len(reduced),
            'same_answer_controls':len(controls),'minimum_shape_iou':min_iou,
            'all_pixel_locality_checks':True,'all_native_truth_checks':True,'all_reciprocal_checks':True,
            'source_components_disjoint':True,'structural':structural,
            'limitations':['single-object appearance, not texture-invariant pure color',
                          'counts and truth visually reviewed by assistant, not independent human annotation',
                          'source pool partly exposed to earlier Qwen runs','class prior is not uniform',
                          'model correctness and token/hook calibration remain untested']}
    if finalize:
        receipt=json.loads((root/'visual_review.json').read_text())
        if receipt.get('status')!='passed' or set(receipt['source_ids'])!={r['source_id'] for r in edits}:
            raise ValueError('complete visual review receipt required')
        for i in items:i['review_status']='passed_assistant_visual_and_automated_review'
        manifest=expand(root,items,seed=SEED)
        summary=json.loads((root/'build_summary.json').read_text())
        summary.update(dataset_hash=manifest['dataset_hash'],visual_review='passed_assistant_review',
                       construction_status='passed',scientific_readiness=False)
        dump(root/'build_summary.json',summary)
        for r in edits:r['review_status']='passed_assistant_visual_and_automated_review'
        for r in controls:r['review_status']='passed_assistant_visual_and_automated_review'
        dump(root/'edits.json',edits);dump(root/'same_answer_controls.json',controls)
    dump(root/'independent_audit.json',report)
    files={str(p.relative_to(root)).replace('\\','/'):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='release_manifest.json'}
    dump(root/'release_manifest.json',{'files':files,'release_sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()})
    print(json.dumps(report,indent=2))


def control_sheets(root):
    controls=json.loads((root/'same_answer_controls.json').read_text())
    edits={r['pair_id']:r for r in json.loads((root/'edits.json').read_text())}
    for start in range(0,len(controls),32):
        page=Image.new('RGB',(1000,8*100),'white');draw=ImageDraw.Draw(page)
        for k,c in enumerate(controls[start:start+32]):
            x=(k%4)*250;y=(k//4)*100;r=edits[c['pair_id']]
            draw.text((x+2,y+1),f"{start+k:03} {r['source_id']} same {c['answer']}",fill='black')
            j=c['changed_object'];box=r['objects_before'][j]['bbox'];w,h=r['total'],r['black_before']
            original=Image.open(root/'native'/f"{c['pair_id']}_x.png").convert('RGB')
            before=original.crop(box);before.thumbnail((100,70));page.paste(before,(x,y+25))
            control=Image.open(root/c['image']);control.thumbnail((140,70));page.paste(control,(x+105,y+25))
        page.save(root/'review'/f'controls_{start:03}.png')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--finalize-review',action='store_true')
    p.add_argument('--control-sheets',action='store_true');args=p.parse_args()
    if args.control_sheets:control_sheets(args.root)
    else:audit(args.root,args.finalize_review)
