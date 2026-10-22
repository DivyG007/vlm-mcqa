"""Reuse audited candidate pools; find exact native one-object replacements."""
import hashlib,json
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from PIL import Image
from probe_iconqa_marbles import objects,answer

AUDIT=Path('D:/college/Sem 5/ANLP/Project/ANLP-project/results/iconqa_eda_20261026')
RAW=Path('D:/college/Sem 5/ANLP/Project/Dataset/iconqa_data/iconqa_data/iconqa')
OUT=Path('results/iconqa_cf_build_20261029')


def main():
    OUT.mkdir(exist_ok=True)
    wanted={}
    for split in ('train','val','test'):
        for line in (AUDIT/'supplement_final'/f'quarantined_review_pool_{split}.jsonl').open(encoding='utf-8-sig'):
            p=json.loads(line)
            if set(p.get('options_x',[]))=={'certain','impossible','probable','unlikely'} or p.get('skills')==['probability']:
                wanted[str(p['x'])]=split;wanted[str(p['y'])]=split
    inventory={}
    for line in (AUDIT/'inventory.jsonl').open(encoding='utf-8-sig'):
        r=json.loads(line)
        if str(r['pid']) in wanted:inventory[str(r['pid'])]=r
    records=[];excluded=[];groups=defaultdict(list)
    for pid in sorted(wanted):
        split=wanted[pid]; r=inventory[pid];d=r['data'];path=RAW/split/'choose_txt'/pid/'image.png'
        try:
            im=Image.open(path).convert('RGB');a=np.array(im)
            if hashlib.sha256(path.read_bytes()).hexdigest()!=r['byte_sha256']:raise ValueError('source hash changed')
            obs=objects(im);black=sum(o['color']=='black' for o in obs);truth=answer(black,len(obs))
            if truth!=d['choices'][d['answer']]:raise ValueError('native count/label disagreement')
            overlaps=False
            for i,o in enumerate(obs):
                for other in obs[i+1:]:
                    x,y,X,Y=o['bbox'];u,v,U,V=other['bbox']
                    if max(x,u)<min(X,U) and max(y,v)<min(Y,V):overlaps=True
            changes=[]
            for i,o in enumerate(obs):
                new=black+(1 if o['color']!='black' else -1)
                try: cf=answer(new,len(obs))
                except ValueError:continue
                if cf!=truth and not overlaps:changes.append({'object_index':i,'new_answer':cf,'black_after':new})
            rec={'pid':pid,'source_split':split,'path':str(path),'question':d['question'],'grade':d['grade'],
                 'label':d['label'],'skills':r['skills'],'source_sha256':r['byte_sha256'],
                 'source_pixel_sha256':r['pixel_sha256'],'source_choices':d['choices'],
                 'source_answer_index':d['answer'],'size':list(im.size),'objects':obs,
                 'black':black,'total':len(obs),'answer':truth,'possible_edits':changes,
                 'overlapping_boxes':overlaps}
            records.append(rec)
            fingerprints=[]
            for o in obs:
                x,y,X,Y=o['bbox']; fingerprints.append((tuple(o['bbox']),hashlib.sha256(a[y:Y,x:X].tobytes()).hexdigest()))
            # Match all UNCHANGED marbles exactly; edited sprite bounds may differ.
            for j in range(len(obs)):
                key=(split,d['question'],im.size,len(obs),tuple(sorted(fingerprints[:j]+fingerprints[j+1:])))
                groups[key].append(rec)
        except ValueError as e:excluded.append({'pid':pid,'split':split,'reason':str(e)})
    pairs=[];seen=set()
    for group in groups.values():
        if len(group)<2:continue
        images={r['pid']:np.array(Image.open(r['path']).convert('RGB')) for r in group}
        for i,x in enumerate(group):
            for y in group[i+1:]:
                edge=tuple(sorted((x['pid'],y['pid'])))
                if edge in seen or x['answer']==y['answer']:continue
                delta=np.any(images[x['pid']]!=images[y['pid']],axis=2)
                for j,o in enumerate(x['objects']):
                    a,b,A,B=o['bbox'];outside=delta.copy();outside[b:B,a:A]=False
                    if delta.any() and not outside.any() and abs(x['black']-y['black'])==1:
                        pairs.append({'x':x['pid'],'y':y['pid'],'split':x['source_split'],
                                      'object_index':j,'bbox':o['bbox'],'changed_pixels':int(delta.sum()),
                                      'answers':[x['answer'],y['answer']]});seen.add(edge);break
    data={'source':'audited quarantined probability pool','records':records,'excluded':excluded,'native_exact_pairs':pairs}
    (OUT/'candidate_inspection_v2.json').write_text(json.dumps(data,indent=2))
    print(json.dumps({'audited_sources':dict(Counter(wanted.values())),
                      'valid_sources':dict(Counter(r['source_split'] for r in records)),
                      'one_edit_capable':dict(Counter(r['source_split'] for r in records if r['possible_edits'])),
                      'native_exact_pair_edges':dict(Counter(p['split'] for p in pairs)),
                      'exclusions':dict(Counter(r['reason'] for r in excluded)),
                      'layout_groups_repeated':sum(len(v)>1 for v in groups.values())},indent=2))


if __name__=='__main__':main()
