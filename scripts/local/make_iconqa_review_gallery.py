"""Self-contained image review; does not alter the frozen scientific release."""
import base64,json,re
from pathlib import Path


def main():
    root=Path('data/generated/iconqa_cf_core7_small_v1_20261029_attempt2')
    out=Path('results/iconqa_qwen_small_preparation_20261029')
    md=out/'review_all_pairs.md'
    md.write_text(re.sub(r'(!\[[^\]]*\])\(([^<>\n]+)\)',r'\1(<\2>)',md.read_text(encoding='utf-8')),encoding='utf-8')
    pairs=json.loads((root/'edits.json').read_text())
    for r in pairs:
        for member in ('x','y'):
            data=(root/'images'/f"{r['pair_id']}_{member}.png").read_bytes()
            r[member]='data:image/png;base64,'+base64.b64encode(data).decode()
    data=[{k:r[k] for k in ('source_id','source_split','native_answer','edited_answer','black_before','black_after','total','question','x','y')} for r in pairs]
    html='''<!doctype html><html><head><meta charset="utf-8"><title>IconQA pair review</title>
<style>body{font:17px system-ui;background:#f3f5f8;color:#172435;margin:0}main{max-width:1100px;margin:auto;padding:24px}h1{margin:0 0 10px}nav{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:20px 0}button,select{font:inherit;padding:9px 14px;border:1px solid #aab7c5;border-radius:7px;background:white;cursor:pointer}#pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}figure{margin:0;background:white;padding:16px;border:1px solid #ccd5df;border-radius:10px}img{display:block;width:100%;height:auto;max-width:448px;margin:auto}figcaption{font-weight:650;margin-bottom:12px}.answer{margin:12px 0 0}#meta{font-size:14px;color:#506174}#question{font-size:20px;line-height:1.5}p{line-height:1.5}#status{font-weight:650}@media(max-width:650px){#pair{grid-template-columns:1fr}}</style></head>
<body><main><h1>Native IconQA / edited companion</h1><p>Review the marble outline, size, shading and texture. One marble appearance is intentionally changed. All images are embedded in this file; no external paths are required.</p>
<nav><button id="prev">← Previous</button><span id="status"></span><button id="next">Next →</button><label>Pair <select id="choose"></select></label><button id="hide">Hide identity labels</button></nav>
<p id="meta"></p><p id="question"></p><section id="pair"><figure><figcaption id="leftLabel"></figcaption><img id="left" alt="Left marble diagram"><p class="answer" id="leftAnswer"></p></figure><figure><figcaption id="rightLabel"></figcaption><img id="right" alt="Right marble diagram"><p class="answer" id="rightAnswer"></p></figure></section>
<p>Use the arrow keys or dropdown to inspect all 136 pairs. The release was checked automatically and reviewed by the assistant; this gallery is for your own visual verification.</p></main>
<script>const pairs=DATA;let index=0,blind=false;const $=id=>document.getElementById(id);pairs.forEach((p,i)=>{let o=document.createElement('option');o.value=i;o.textContent=(i+1)+' — '+p.source_id+' ('+p.source_split+')';$('choose').append(o)});function show(){const p=pairs[index];const swap=blind&&index%2===0;const left=swap?'y':'x',right=swap?'x':'y';$('left').src=p[left];$('right').src=p[right];$('leftLabel').textContent=blind?'Image A':'Native IconQA';$('rightLabel').textContent=blind?'Image B':'Edited companion';$('leftAnswer').textContent=blind?'':p.native_answer+' — '+p.black_before+' black / '+p.total;$('rightAnswer').textContent=blind?'':p.edited_answer+' — '+p.black_after+' black / '+p.total;$('status').textContent=(index+1)+' / '+pairs.length;$('meta').textContent='Source ID '+p.source_id+' · Original split '+p.source_split;$('question').textContent=p.question;$('choose').value=index;$('prev').disabled=index===0;$('next').disabled=index===pairs.length-1;$('hide').textContent=blind?'Reveal identity labels':'Hide identity labels'}$('prev').onclick=()=>{if(index>0){index--;show()}};$('next').onclick=()=>{if(index<pairs.length-1){index++;show()}};$('choose').onchange=e=>{index=Number(e.target.value);show()};$('hide').onclick=()=>{blind=!blind;show()};document.addEventListener('keydown',e=>{if(e.target.tagName==='SELECT')return;if(e.key==='ArrowRight')$('next').click();if(e.key==='ArrowLeft')$('prev').click()});show();</script></body></html>'''
    html=html.replace('DATA',json.dumps(data).replace('</','<\\/'))
    path=out/'review_pairs.html';path.write_text(html,encoding='utf-8')
    print(json.dumps({'gallery':str(path),'pairs':len(data),'embedded_images':len(data)*2,'bytes':path.stat().st_size}))


if __name__=='__main__':main()
