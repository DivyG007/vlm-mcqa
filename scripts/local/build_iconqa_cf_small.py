"""Deterministic, source-preserving native-marble appearance counterfactuals."""
import argparse,hashlib,json,random
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from PIL import Image,ImageOps,ImageDraw
from probe_iconqa_marbles import objects,answer
from vlm_mcqa.clevr_mcq4.build import expand

SEED=20261029
CHOICES=['certain','impossible','probable','unlikely']
PHASE={'train':'calibration','val':'discovery','test':'confirmation'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,data):Path(path).write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
def pixels(im):return hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
def rank(text):return hashlib.sha256(f'{SEED}:{text}'.encode()).hexdigest()


def replace(image,obs,index,sprite,expected):
    box=obs[index]['bbox'];target=sprite.resize((box[2]-box[0],box[3]-box[1]),Image.Resampling.BICUBIC)
    a=np.array(image); b=a.copy();b[box[1]:box[3],box[0]:box[2]]=np.array(target)
    result=Image.fromarray(b);delta=np.any(a!=b,axis=2)
    if not delta.any():raise ValueError('identical sprite')
    mask_a=a[box[1]:box[3],box[0]:box[2]].min(2)<240
    mask_b=np.array(target).min(2)<240
    iou=float((mask_a&mask_b).sum()/max(1,(mask_a|mask_b).sum()))
    if iou<.94:raise ValueError('sprite shape mismatch')
    changed=objects(result)
    if len(changed)!=len(obs):raise ValueError('object count changed')
    for k,o in enumerate(changed):
        if k!=index and (o['bbox']!=obs[k]['bbox'] or o['color']!=obs[k]['color']):
            raise ValueError('other object changed')
    black=sum(o['color']=='black' for o in changed)
    if answer(black,len(changed))!=expected:raise ValueError('edited truth mismatch')
    outside=delta.copy();outside[box[1]:box[3],box[0]:box[2]]=False
    if outside.any():raise ValueError('nonlocal change')
    return result,Image.fromarray(delta.astype('uint8')*255),{'changed_pixels':int(delta.sum()),
            'shape_iou':iou,'black_after':black,'objects_after':changed}


def letterbox(im):
    fitted=ImageOps.contain(im,(448,448),Image.Resampling.BICUBIC)
    result=Image.new('RGB',(448,448),'white');offset=((448-fitted.width)//2,(448-fitted.height)//2)
    result.paste(fitted,offset)
    return result,{'source_size':list(im.size),'resized_size':list(fitted.size),'offset_xy':list(offset),
                   'canvas':[448,448],'interpolation':'bicubic'}


def choose_directions(contrasts):
    """Select2 reciprocal relationships per kind; alternate by pair to cover positions."""
    grouped=defaultdict(list)
    for c in contrasts:grouped[c['pair_id']].append(c)
    out=[];phase_index=Counter()
    for pair,cs in sorted(grouped.items()):
        split=cs[0]['split'];index=phase_index[split];phase_index[split]+=1
        for kind in ('position','content','content_and_position','identity'):
            pool=[c for c in cs if c['contrast_type']==kind]
            # Native factorial generator need not emit reciprocal pairs; construct
            # reciprocal-valid factorial manifest upstream before reducing it.
            edges={ (c['donor_variant_id'],c['recipient_variant_id']):c for c in pool}
            groups=[]
            for (a,b),c in edges.items():
                if a<b and (b,a) in edges:groups.append((rank(pair+kind+a+b),[c,edges[b,a]]))
            groups=sorted(groups)
            if kind=='position':
                chosen=[]
                for member,p in [('x',index%4),('y',(index+2)%4)]:
                    q=(p+1)%4
                    matches=[g for _,g in groups if ('_'+member+'|') in g[0]['donor_variant_id']
                             and {g[0]['donor_symbol'],g[0]['recipient_symbol']}=={('A','B','C','D')[p],('A','B','C','D')[q]}]
                    chosen.append(matches[0])
            elif kind in ('content','identity'):
                chosen=[]
                for k in range(2):
                    pos=(index+(k if kind=='content' else 2*k))%4
                    member='x' if k==0 else 'y'
                    matches=[g for _,g in groups if g[0]['donor_symbol']==('A','B','C','D')[pos]
                             and (kind=='content' or ('_'+member+'|') in g[0]['donor_variant_id'])]
                    chosen.append(matches[0])
            else:
                choices=[]
                for j,(key,g) in enumerate(groups):
                    for key2,h in groups[j+1:]:
                        if len({g[0]['donor_symbol'],g[0]['recipient_symbol'],h[0]['donor_symbol'],h[0]['recipient_symbol']})==4:
                            choices.append((rank(pair+key+key2),[g,h]))
                chosen=min(choices,key=lambda x:x[0])[1] if choices else []
            selected=[c for g in chosen for c in g]
            if len(selected)!=4:raise ValueError('insufficient reciprocal '+kind)
            out.extend(selected)
        out.extend([c for c in cs if c['contrast_type']=='random_pair'])
    return out


def build(args):
    inspection=json.loads(args.inspection.read_text())
    records=inspection['records'];train_ids=set(json.loads(args.train_review.read_text())['item_ids'])
    templates=[]
    for r in records:
        if 'iconqa_'+r['pid'] not in train_ids:continue
        im=Image.open(r['path']).convert('RGB')
        for j,o in enumerate(r['objects']):
            templates.append({'source_id':r['pid'],'index':j,'color':o['color'],'box':o['bbox'],'image':im.crop(o['bbox'])})
    if not {t['color'] for t in templates}=={'black','nonblack'}:raise ValueError('training sprites unavailable')
    if args.output.exists():raise ValueError('fresh output required')
    args.output.mkdir(parents=True)
    for folder in ('images','native','masks','controls','review'):(args.output/folder).mkdir()
    selected=[];rejected=[];seen=set();totals={}
    for split,need in [('train',8),('val',64),('test',64)]:
        buckets={a:sorted([r for r in records if r['source_split']==split and r['answer']==a and r['possible_edits']],
                         key=lambda r:rank(split+r['pid'])) for a in CHOICES}
        accepted=[]
        while any(buckets.values()) and len(accepted)<need:
            for label in CHOICES:
                if not buckets[label] or len(accepted)>=need:continue
                r=buckets[label].pop(0);im=Image.open(r['path']).convert('RGB');success=None
                if sha(r['path'])!=r['source_sha256']:raise ValueError('source changed')
                for edit in sorted(r['possible_edits'],key=lambda e:rank(r['pid']+str(e))):
                    idx=edit['object_index'];color='nonblack' if r['objects'][idx]['color']=='black' else 'black'
                    own=[{'source_id':r['pid'],'index':j,'color':o['color'],'box':o['bbox'],'image':im.crop(o['bbox'])}
                         for j,o in enumerate(r['objects']) if o['color']==color]
                    choices=own+sorted([t for t in templates if t['color']==color],key=lambda t:rank(r['pid']+t['source_id']+str(t['index'])))
                    for t in choices:
                        try:cf,mask,info=replace(im,r['objects'],idx,t['image'],edit['new_answer'])
                        except ValueError:continue
                        native_hash=pixels(im);cf_hash=pixels(cf)
                        if native_hash in seen or cf_hash in seen:continue
                        if pixels(im.copy())!=native_hash:raise ValueError('noop failure')
                        success=(edit,t,cf,mask,info);break
                    if success:break
                if not success:
                    rejected.append({'pid':r['pid'],'split':split,'reason':'no valid nonduplicate one-object replacement'});continue
                edit,t,cf,mask,info=success;seen.update((pixels(im),pixels(cf)))
                accepted.append((r,edit,t,im,cf,mask,info))
        if len(accepted)!=need:raise ValueError(f'{split} only {len(accepted)} valid pairs, need {need}')
        selected.extend(accepted);totals[split]=len(accepted)
    items=[];edits=[];controls=[];images={}
    for r,e,t,im,cf,mask,info in selected:
        pair=f"iconqa_cf_{r['source_split']}_{r['pid']}";split=PHASE[r['source_split']]
        im.save(args.output/'native'/f'{pair}_x.png');cf.save(args.output/'native'/f'{pair}_y.png')
        mask.save(args.output/'masks'/f'{pair}.png')
        editrec={'pair_id':pair,'source_id':r['pid'],'source_split':r['source_split'],'source_sha256':r['source_sha256'],
                 'source_pixel_sha256':r['source_pixel_sha256'],'source_path':r['path'],'question':r['question'],
                 'objects_before':r['objects'],'black_before':r['black'],'total':r['total'],
                 'native_answer':r['answer'],'edited_answer':e['new_answer'],'changed_object':e['object_index'],
                 'edit_bbox':r['objects'][e['object_index']]['bbox'],'sprite_source_id':t['source_id'],
                 'sprite_object_index':t['index'],'sprite_bbox':t['box'],'sprite_split':'train' if t['source_id']!=r['pid'] else r['source_split'],
                 'edit_type':'single_object_appearance_replacement','pure_color_edit':False,
                 'unchanged_outside_bbox':True,'noop_pixel_equal':True,'review_status':'pending',**info}
        edits.append(editrec)
        for member,image,truth in [('x',im,r['answer']),('y',cf,e['new_answer'])]:
            processed,geometry=letterbox(image);relative=f'images/{pair}_{member}.png';processed.save(args.output/relative)
            images[relative]=sha(args.output/relative)
            items.append({'item_id':pair+'_'+member,'source':'iconqa_derived','dataset_version':'iconqa_cf_core7_small_v1',
                          'split':split,'scene_id':r['source_pixel_sha256'],'counterfactual_pair_id':pair,'pair_member':member,
                          'pair_provenance':'verified_single_marble_appearance_edit','image':relative,'question':r['question'],
                          'option_contents':CHOICES,'semantic_answer':truth,'answer_type':'text',
                          'question_family':r['grade']+'_'+r['label'],'program':[],'program_depth':None,
                          'answer_rule':{'operation':'black_marble_probability','black':r['black'] if member=='x' else info['black_after'],'total':r['total']},
                          'source_id':r['pid'],'source_split':r['source_split'],'skills':r['skills'],
                          'source_choices':r['source_choices'],'source_answer_index':r['source_answer_index'],
                          'source_image_sha256':r['source_sha256'],'geometry':geometry,
                          'image_status':'native' if member=='x' else 'edited','review_status':'pending',
                          'prior_qwen_exposure':'source_pool_partly_used; no pristine-source holdout claim'})
        # Same-answer visual-edit control, never an independent data pair.
        for j,o in enumerate(r['objects']):
            done=False
            for t2 in templates:
                if t2['color']!=o['color']:continue
                try:control,cmask,cinfo=replace(im,r['objects'],j,t2['image'],r['answer'])
                except ValueError:continue
                path=f'controls/{pair}.png';letterbox(control)[0].save(args.output/path)
                controls.append({'pair_id':pair,'image':path,'answer':r['answer'],'review_status':'pending',
                                 'source_split':r['source_split'],'sprite_source_id':t2['source_id'],
                                 'changed_object':j,**cinfo});done=True;break
            if done:break
    manifest=expand(args.output,items,seed=SEED)
    # Complete factorial reciprocity explicitly, respecting distinct readout labels.
    contrasts=[json.loads(s) for s in (args.output/'contrasts.jsonl').read_text().splitlines()]
    from vlm_mcqa.clevr_mcq4.variants import _contrast
    variants={v['variant_id']:v for v in (json.loads(s) for s in (args.output/'variants.jsonl').read_text().splitlines())}
    factorial=[]
    per_pair=defaultdict(lambda:defaultdict(list))
    for v in variants.values():
        if v['label_scheme']=='letters' and v['variant_kind']=='position':
            per_pair[v['counterfactual_pair_id']][v['pair_member']].append(v)
    for members in per_pair.values():
        for a in members['x']:
            for b in members['y']:
                pos=b['option_contents'].index(a['semantic_answer'])
                reverse=a['option_contents'].index(b['semantic_answer'])
                if len({a['correct_index'],b['correct_index'],pos})==3 and len({a['correct_index'],b['correct_index'],reverse})==3:
                    factorial.extend([_contrast('content_and_position',a,b,content_index=pos,content_symbol=b['labels'][pos]),
                                      _contrast('content_and_position',b,a,content_index=reverse,content_symbol=a['labels'][reverse])])
    full=[c for c in contrasts if c['contrast_type']!='content_and_position']+factorial
    # Both-directions factorial constraints can restrict reciprocal edges. Keep
    # separate manifest, and fail rather than emit an asymmetric reduced sample.
    reduced=choose_directions(full)
    dump(args.output/'reduced_contrasts.json',reduced)
    dump(args.output/'edits.json',edits);dump(args.output/'same_answer_controls.json',controls)
    dump(args.output/'construction_exclusions.json',rejected)
    summary={'dataset_hash':manifest['dataset_hash'],'pairs_by_source_split':totals,'items':len(items),
             'primary_variants':len(items)*12,'all_variants':len(variants),'reduced_contrasts':len(reduced),
             'native_answer_counts':{s:dict(Counter(r['answer'] for r,*_ in selected if r['source_split']==s)) for s in PHASE},
             'semantic_answer_counts':dict(Counter(i['semantic_answer'] for i in items)),
             'transitions':dict(Counter(r['native_answer']+'->'+r['edited_answer'] for r in edits)),
             'auxiliary_same_answer_images':len(controls),'native_exact_pairs_found':len(inspection['native_exact_pairs']),
             'seed':SEED,'visual_review':'pending','model_eligibility':'not_tested','scientific_readiness':False,
             'inspection_sha256':sha(args.inspection),'builder_sha256':sha(__file__),
             'reduced_profile':{'pairs':32,'behavior_items':128,'head_screen_contrasts':64,'head_candidates':16,
                                'random_sets':20,'ablation_recipients':32,'batch_size':'requires model pilot'}}
    dump(args.output/'build_summary.json',summary)
    # Paged review, factual/edited shown side-by-side, native resolution before letterbox.
    for start in range(0,len(edits),16):
        page=Image.new('RGB',(1200,4*190),'white');draw=ImageDraw.Draw(page)
        for k,r in enumerate(edits[start:start+16]):
            x=(k%4)*300;y=(k//4)*190
            draw.text((x+2,y+2),f"{start+k:03} {r['source_id']} B{r['black_before']}->{r['black_after']}/N{r['total']}",fill='black')
            draw.text((x+2,y+17),r['native_answer']+' -> '+r['edited_answer'],fill='black')
            for j,member in enumerate(('x','y')):
                thumb=Image.open(args.output/'native'/f"{r['pair_id']}_{member}.png");thumb.thumbnail((145,150))
                page.paste(thumb,(x+j*150,y+35))
        page.save(args.output/'review'/f'pairs_{start:03}.png')
    files={str(p.relative_to(args.output)).replace('\\','/'):sha(p) for p in args.output.rglob('*') if p.is_file()}
    dump(args.output/'release_manifest.json',{'files':files,'release_sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inspection',type=Path,required=True);p.add_argument('--train-review',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);build(p.parse_args())
