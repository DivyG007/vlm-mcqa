"""Strict release and independent-pair sampler for the small IconQA protocol."""
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path
import random

DATASET_HASH='371ebbd9278cc3ba'
RELEASE_HASH='93efc68344571dda71999086df9b4638294220bce19901eb11998f519931c87c'
SEED=20261029
EXPECTED={'position':4,'content':4,'content_and_position':4,'identity':4,'random_pair':2}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_release(root):
    root=Path(root).resolve()
    release=json.loads((root/'release_manifest.json').read_text())
    actual=hashlib.sha256(json.dumps(release['files'],sort_keys=True).encode()).hexdigest()
    if actual!=RELEASE_HASH or release['release_sha256']!=actual:raise ValueError('small release binding changed')
    for name,sha in release['files'].items():
        path=(root/name).resolve()
        if root not in path.parents or not path.is_file() or digest(path)!=sha:
            raise ValueError('small release file changed: '+name)
    manifest=json.loads((root/'dataset_manifest.json').read_text())
    computed=hashlib.sha256(json.dumps(manifest['sha256'],sort_keys=True).encode()).hexdigest()[:16]
    if manifest['dataset_hash']!=DATASET_HASH or computed!=DATASET_HASH:raise ValueError('small dataset identity changed')
    audit=json.loads((root/'independent_audit.json').read_text())
    review=json.loads((root/'visual_review.json').read_text())
    if audit.get('status')!='passed' or review.get('status')!='passed':raise ValueError('construction review gate failed')
    items=[json.loads(s) for s in (root/'items.jsonl').read_text().splitlines()]
    if Counter(i['split'] for i in items)!={'calibration':16,'discovery':128,'confirmation':128}:
        raise ValueError('small phase sizes changed')
    if set(review['source_ids'])!={i['source_id'] for i in items}:raise ValueError('incomplete visual review')
    return manifest,release


def select_pairs(session,phase,contrasts,target=32):
    by_pair=defaultdict(list)
    for c in contrasts:
        if c['split']==phase:by_pair[c['pair_id']].append(c)
    ordered=sorted(by_pair,key=lambda p:hashlib.sha256(f'{SEED}:{phase}:{p}'.encode()).hexdigest())
    chosen=[];scans=[]
    for pair in ordered:
        cs=by_pair[pair]
        if Counter(c['contrast_type'] for c in cs)!=EXPECTED:raise ValueError('reduced per-pair counts changed')
        needed={c[k] for c in cs if c['contrast_type']!='random_pair' for k in ('donor_variant_id','recipient_variant_id')}
        # Random/ablation donors can use any position of a selected pair.
        needed.update(v['variant_id'] for v in session.variants.values()
                      if v['counterfactual_pair_id']==pair and v['label_scheme']=='letters' and v['variant_kind']=='position')
        checks=[]
        for variant in sorted(needed):
            if session.variants[variant]['split']!=phase:raise ValueError('phase leak')
            clean=session.clean(variant)
            checks.append({'variant_id':variant,'correct':bool(clean.correct),'margin':clean.readout.get('margin')})
        eligible=all(c['correct'] for c in checks)
        scans.append({'pair_id':pair,'eligible':eligible,'checks':checks})
        if eligible:chosen.append(pair)
        if len(chosen)==target:break
    coverage={'target_pairs':target,'pairs_used':len(chosen),'pair_ids':chosen,'scanned_pairs':len(scans),
              'selection_rule':'fixed hash order; all retained and potential unrelated/ablation position variants correct',
              'checks':scans}
    if len(chosen)!=target:return [],coverage,[],[]
    selected=set(chosen)
    reduced=[c for c in contrasts if c['pair_id'] in selected and c['split']==phase and c['contrast_type']!='random_pair']
    from vlm_mcqa.clevr_mcq4.variants import build_random_pair_contrasts
    pv={p:[v for v in session.variants.values() if v['counterfactual_pair_id']==p] for p in chosen}
    reduced+=build_random_pair_contrasts(pv,random.Random(SEED))
    head=[];recipients=[]
    for index,pair in enumerate(chosen):
        member='x' if index%2==0 else 'y'
        positions=[c for c in reduced if c['pair_id']==pair and c['contrast_type']=='position'
                   and ('_'+member+'|') in c['donor_variant_id']]
        if len(positions)!=2:raise ValueError('head screen needs one reciprocal native/edited relationship')
        head.extend(positions)
        recipients.append(positions[0]['recipient_variant_id'])
    if Counter(c['contrast_type'] for c in reduced)!={k:v*target for k,v in EXPECTED.items()}:
        raise ValueError('small selection work counts changed')
    coverage.update(cases_by_type=dict(Counter(c['contrast_type'] for c in reduced)),
                    head_cases=len(head),ablation_recipients=len(recipients),
                    ablation_source_units=len({session.variants[v]['counterfactual_pair_id'] for v in recipients}))
    return reduced,coverage,head,recipients
