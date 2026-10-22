"""Train-only object extraction feasibility probe, not a dataset release."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage


def objects(image):
    a=np.array(image.convert('RGB'))
    labels,n=ndimage.label((a.min(2)<240))
    result=[]
    for i,sl in enumerate(ndimage.find_objects(labels),1):
        if sl is None: continue
        mask=labels[sl]==i; area=int(mask.sum())
        if area<20: continue
        y,x=sl; w=x.stop-x.start; h=y.stop-y.start
        if min(w,h)<5 or not .75<w/h<1.3 or not .5<area/(w*h)<.9:
            raise ValueError('non-marble component')
        pix=a[sl][mask].astype(float)
        saturation=float(np.mean(pix.max(1)-pix.min(1)))
        if not (saturation<5 or saturation>20): raise ValueError('ambiguous color')
        result.append({'bbox':[x.start,y.start,x.stop,y.stop],'color':'nonblack' if saturation>20 else 'black',
                       'area':area,'saturation':saturation,'mean_rgb':pix.mean(0).tolist()})
    if not result: raise ValueError('empty')
    return result


def answer(black,total):
    if black==0:return 'impossible'
    if black==total:return 'certain'
    if 2*black==total:raise ValueError('equal probability boundary excluded')
    return 'probable' if 2*black>total else 'unlikely'


if __name__=='__main__':
    root=Path('data/generated/iconqa_probability_mcq4_v1_20261028')
    raw=Path('D:/college/Sem 5/ANLP/Project/Dataset/iconqa_data/iconqa_data/iconqa')
    items=[json.loads(s) for s in (root/'items.jsonl').read_text().splitlines()]
    train=[i for i in items if i['split']=='calibration']
    out=Path('results/iconqa_cf_build_20261029');out.mkdir(exist_ok=True)
    sheet=Image.new('RGB',(1000,4*220),'white');draw=ImageDraw.Draw(sheet);records=[]
    for k,i in enumerate(train):
        path=raw/i['source_split']/'choose_txt'/i['source_id']/'image.png'
        im=Image.open(path).convert('RGB');obs=objects(im)
        black=sum(o['color']=='black' for o in obs); predicted=answer(black,len(obs))
        records.append({'source_id':i['source_id'],'objects':obs,'native_answer':i['semantic_answer'],
                        'count_answer':predicted,'matches':predicted==i['semantic_answer']})
        thumb=im.copy();d=ImageDraw.Draw(thumb)
        for o in obs:d.rectangle(o['bbox'],outline='red',width=1)
        thumb.thumbnail((240,180)); x=(k%4)*250;y=(k//4)*220
        sheet.paste(thumb,(x,y+25));draw.text((x+2,y+2),f"{i['source_id']} B{black}/N{len(obs)} {predicted}",fill='black')
    sheet.save(out/'train_objects.png');(out/'train_probe.json').write_text(json.dumps(records,indent=2))
    print(json.dumps({'items':len(records),'truth_matches':sum(r['matches'] for r in records)}))
