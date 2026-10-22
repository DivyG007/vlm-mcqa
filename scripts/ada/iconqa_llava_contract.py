"""CPU-readable input and pilot contracts for the prepared LLaVA campaign."""
import json
from pathlib import Path
from iconqa_small_inputs import digest,RELEASE_HASH
from stage_iconqa_transfer import MODELS


def verify_checkpoint(root):
    root=Path(root);receipt=json.loads((root/'verified_checkpoint.json').read_text())
    model,revision=MODELS['llava_ov_7b_si']
    if (receipt.get('model_id'),receipt.get('revision'),receipt.get('status'))!=(model,revision,'verified'):
        raise ValueError('pinned publisher-verified LLaVA required')
    for name,r in receipt['files'].items():
        f=root/name
        if f.stat().st_size!=r['bytes'] or (not name.endswith('.safetensors') and digest(f)!=r['sha256']):
            raise ValueError('checkpoint changed '+name)


def verify_pilot(root):
    root=Path(root);gate_path=Path(__file__).resolve().parents[2]/'configs/core7/iconqa_llava_pilot_gate.json'
    gate=json.loads(gate_path.read_text())
    for name,sha in gate['files'].items():
        if digest(root/name)!=sha:raise ValueError('pilot evidence changed '+name)
    status=json.loads((root/'status.json').read_text());summary=json.loads((root/'pilot_summary.json').read_text())
    calibration=json.loads((root/'calibration.json').read_text());records=json.loads((root/'batch_benchmark.json').read_text())['records']
    model,revision=MODELS['llava_ov_7b_si'];spec=status['model_spec']
    from iconqa_transfer_pilot import placement_for
    expected,_=placement_for('llava_ov_7b_si',28)
    if (status['status'],status['model_id'],status['revision'],status['release'],status['job_id'])!=('complete',model,revision,RELEASE_HASH,gate['job_id']):
        raise ValueError('wrong completed pilot identity')
    if spec['dtype']!='float16' or spec['attn_implementation']!='sdpa' or spec['device_map']!=expected or spec['processor_kwargs']!={}:
        raise ValueError('pilot processor/precision/placement mismatch')
    batch2=[r for r in records if r['batch_size']==2]
    cells={(r['scheme'],r['image_status']) for r in batch2}
    if len(records)!=18 or len(batch2)!=6 or cells!={(s,i) for s in ('letters','numbers','rare_letters') for i in ('native','edited')}:
        raise ValueError('incomplete pilot coverage')
    if not calibration['all_passed'] or any(not r['equivalence_passed'] for r in records if r['batch_size'] in (1,2)):
        raise ValueError('pilot did not admit batch2')
    if {r['image_tokens'] for r in records}!={3699}:raise ValueError('unexpected pilot geometry')
    policy=status.get('sdpa_policy',{})
    if policy.get('math_fallback') is not False or policy.get('gqa')!='explicit repeated KV':raise ValueError('unadmitted attention policy')
    return {'max_tokens':max(r['tokens'] for r in records),'sdpa_policy':policy,'sha256':digest(gate_path),'files':gate['files'],'job_id':gate['job_id'],
            'runtime':status['runtime'],'image_tokens':3699,'batch_size':2,'recommended_batch':summary['recommended_batch'],
            'mean_speedup':summary['mean_speedup'],'scope':summary['scope']}


def verify_loaded_placement(adapter,placement):
    if any(str(d) not in ('0','1','cuda:0','cuda:1') for d in adapter.model.hf_device_map.values()):
        raise RuntimeError('CPU/disk offload forbidden')
    for name,param in adapter.model.named_parameters():
        owners=[k for k in placement if name==k or name.startswith(k+'.')]
        if not owners:raise RuntimeError('unmapped parameter '+name)
        expected=placement[max(owners,key=len)]
        if param.device.type!='cuda' or param.device.index!=expected:raise RuntimeError('incorrect parameter placement '+name)
