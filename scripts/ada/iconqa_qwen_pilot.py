"""One real, gated eight-pair train pilot. Never launches discovery/confirmation."""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")


def align_qwen_rotary_inputs(module, args, kwargs):
    """Value-preserving RoPE boundary fix from Rachit's validated Qwen lane."""
    args, kwargs = tuple(args), dict(kwargs)
    hidden = args[0] if args else kwargs.get('x')
    if hidden is None or module.inv_freq.device != hidden.device:
        raise RuntimeError('Qwen rotary hidden/buffer placement mismatch')
    if len(args) >= 2:
        args = (args[0], args[1].to(hidden.device), *args[2:])
    elif 'position_ids' in kwargs:
        kwargs['position_ids'] = kwargs['position_ids'].to(hidden.device)
    return args, kwargs


def verify_data(root):
    manifest = json.loads((root / 'dataset_manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        if digest(root / name) != expected:
            raise ValueError(f'dataset metadata changed: {name}')
    expected_hash = hashlib.sha256(json.dumps(manifest['sha256'], sort_keys=True).encode()).hexdigest()[:16]
    if expected_hash != manifest['dataset_hash']:
        raise ValueError('dataset inventory hash mismatch')
    release = json.loads((root / 'release_manifest.json').read_text())
    expected_release = hashlib.sha256(json.dumps({k:v for k,v in release.items() if k not in ('errors','image_aware_release_hash')},
                                              sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if expected_release != release['image_aware_release_hash'] or release['errors']:
        raise ValueError('image-aware release manifest changed')
    if digest(root / 'build_provenance.json') != release['build_provenance_sha256']:
        raise ValueError('builder provenance changed')
    for name, expected in release['final_image_sha256'].items():
        if digest(root / 'images' / name) != expected:
            raise ValueError(f'image changed: {name}')
    return manifest, release


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, type=Path)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--source-root', required=True, type=Path)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--batch-benchmark', action='store_true')
    args = p.parse_args()
    if not args.execute:
        p.error('real calibration and pilot require --execute')
    if args.output.exists():
        p.error('fresh output required; preserve previous attempts')
    manifest, release = verify_data(args.dataset)
    reviewed = json.loads((args.dataset / 'train_visual_review.json').read_text())
    train_items = [json.loads(line) for line in (args.dataset/'items.jsonl').read_text().splitlines()
                   if json.loads(line)['split']=='calibration']
    if reviewed.get('status') != 'passed' or reviewed.get('dataset_hash') != manifest['dataset_hash']:
        raise ValueError('train-only visual truth/geometry review required')
    if set(reviewed.get('item_ids', [])) != {row['item_id'] for row in train_items}:
        raise ValueError('train visual review membership mismatch')
    receipt_path = args.checkpoint / 'verified_checkpoint.json'
    receipt = json.loads(receipt_path.read_text())
    revision = 'cc594898137f460bfe9f0759e9844b3ce807cfb5'
    if receipt.get('revision') != revision or receipt.get('status') != 'verified':
        raise ValueError('publisher-verified pinned checkpoint required')
    # Check metadata/sizes now; publisher byte hashes were checked once on staging.
    for name, rec in receipt['files'].items():
        path = args.checkpoint / name
        if not path.is_file() or path.stat().st_size != rec['bytes']:
            raise ValueError(f'incomplete checkpoint: {name}')
        if not name.endswith('.safetensors') and digest(path) != rec['sha256']:
            raise ValueError(f'checkpoint metadata changed: {name}')
    args.output.mkdir(parents=True)
    identity = {'stage': 'train_only_technical_pilot', 'status': 'starting', 'scientific_result': False,
                'dataset_hash': manifest['dataset_hash'], 'image_aware_release_hash': release['image_aware_release_hash'],
                'checkpoint_receipt_sha256': digest(receipt_path), 'model_revision': revision,
                'train_review_sha256': digest(args.dataset/'train_visual_review.json'),
                'source_files': {str(path.relative_to(args.source_root)).replace('\\','/'): digest(path)
                                 for path in (args.source_root/'src').rglob('*.py')},
                'pilot_script_sha256': digest(__file__), 'seed': 20261028,
                'job_id': os.environ.get('SLURM_JOB_ID'), 'node': os.environ.get('SLURMD_NODENAME')}
    write(args.output/'status.json', identity)
    try:
        import torch
        import transformers
        from vlm_mcqa.core7.adapters import ModelSpec, VLMAdapter
        from vlm_mcqa.core7.session import Session
        from vlm_mcqa.core7.tracking import Tracker
        from vlm_mcqa.core7.calibration import run_calibration
        from vlm_mcqa.core7.contrasts import select_cases
        from vlm_mcqa.core7.behavior import run_behavior
        from vlm_mcqa.core7.lens import run_lens
        from vlm_mcqa.core7.patching import run_patching
        from vlm_mcqa.core7.components import run_components
        from vlm_mcqa.core7.heads import run_heads
        if torch.cuda.device_count() != 2:
            raise RuntimeError('exactly two allocated GPUs required')
        if any(torch.cuda.get_device_properties(i).total_memory < 10 * 1024**3 for i in range(2)):
            raise RuntimeError('each allocated GPU must have at least 10 GiB')
        if any(torch.cuda.get_device_capability(i) < (7, 5) for i in range(2)):
            raise RuntimeError('pilot requires Turing-or-newer GPUs for the intended FP16 workload')
        placement = {'model.visual': 0, 'model.language_model.embed_tokens': 0,
                     'model.language_model.rotary_emb': 0,
                     'model.language_model.norm': 1, 'lm_head': 1}
        placement.update({f'model.language_model.layers.{i}': 0 if i <= 12 else 1 for i in range(28)})
        config = json.loads((args.checkpoint/'config.json').read_text())
        text = config.get('text_config', config)
        if text.get('num_hidden_layers') != 28:
            raise RuntimeError('unexpected checkpoint architecture')
        spec = ModelSpec(key='qwen25vl_7b', family='qwen_vl', hf_id=str(args.checkpoint),
                         revision=revision, params_b=8.3, lm_backbone='qwen2.5', dtype='float16',
                         device_map=placement, processor_kwargs={'min_pixels':448*448,'max_pixels':448*448})
        started = time.monotonic()
        adapter = VLMAdapter.load(spec)
        load_seconds = time.monotonic() - started
        rotary = adapter.handle.decoder.rotary_emb
        adapter._iconqa_device_hooks = [rotary.register_forward_pre_hook(align_qwen_rotary_inputs, with_kwargs=True)]
        identity.update(torch=torch.__version__, transformers=transformers.__version__,
                        adapter=adapter.describe(), load_seconds=load_seconds,
                        device_boundary='Qwen shared RoPE on GPU0, integer positions routed without value changes',
                        gpu_names=[torch.cuda.get_device_name(i) for i in range(2)])
        write(args.output/'status.json', identity)
        print('[load] checkpoint ready; starting ordinary calibration', flush=True)
        profile = json.loads((args.source_root/'configs/core7/profiles.json').read_text())['profiles']['full_core7']
        # Time actual intervention/control machinery on two balanced directions
        # per train pair; do not repeat an entire science campaign on the pilot.
        # Full scientific targets and global profiles remain unchanged.
        profile.update(pairs=8, behavior_items=16, batch_size=1, bounded_case_preparation=True,
                       contrast_limit_per_type=16, head_screen_contrasts=16, ablation_recipients=16)
        session = Session(adapter, args.dataset, args.output, profile, seed=20261028, prepared_cache=2)
        session.tracker = Tracker()
        # Keep train examples only; even calibration cannot inspect val/test.
        session.items = {k:v for k,v in session.items.items() if v['split']=='calibration'}
        session.variants = {k:v for k,v in session.variants.items() if v['item_id'] in session.items}
        session.contrasts = [v for v in session.contrasts if v['split']=='calibration']
        calibration = run_calibration(session)
        if not calibration['all_passed']:
            raise RuntimeError('ordinary real-model calibration failed; see calibration.json')
        if args.batch_benchmark:
            from iconqa_batch_benchmark import benchmark
            benchmark(session,args.output,identity)
            identity.update(status='complete',benchmark_file='batch_benchmark.json')
            return
        token_audit = []
        for variant in session.variants.values():
            prep = session.prepared(variant['variant_id'])
            contents = [prep.content_ids.get(c) for c in variant['option_contents']]
            if None in contents or len(set(contents)) != 4:
                raise RuntimeError('semantic first-token gate failed')
            if any(not prep.groups.get(g) for g in ('image','question','options','labels')):
                raise RuntimeError('required token group unresolved')
            if len(prep.groups['image']) != 256:
                raise RuntimeError('448px processed image grid is not the expected 256 tokens')
            token_audit.append({'variant_id':variant['variant_id'], 'tokens':prep.num_tokens,
                                'content_ids':contents, 'group_sizes':{k:len(v) for k,v in prep.groups.items()}})
        write(args.output/'token_geometry_audit.json', token_audit)
        # Native heads helper uses discovery semantics; all these IDs are train-only.
        for row in [*session.items.values(), *session.variants.values(), *session.contrasts]:
            row['split'] = 'discovery'
        cases, coverage = select_cases(session, 'discovery', ('position','content','content_and_position'),
                                       max_pairs=8, limit_per_type=None)
        if coverage['pairs_used'] != 8:
            raise RuntimeError(f'train pilot lacks eight eligible pairs: {coverage}')
        write(args.output/'pilot_eligibility.json', coverage)
        del cases
        run = {**identity, 'model_key':'qwen25vl_7b', 'num_layers':adapter.num_layers,
               'profile':profile, 'adapter':adapter.describe(), 'model_spec':dataclasses.asdict(spec),
               'torch':torch.__version__, 'transformers':transformers.__version__, 'load_seconds':load_seconds}
        write(args.output/'run.json', run)
        timings = {}
        functions = {
            'behavior':lambda:run_behavior(session, split='discovery', n_items=16, robust_items=16, compliance_items=16, controls=True),
            'lens':lambda:run_lens(session, split='discovery', n_items=16),
            'patching':lambda:run_patching(session, split='discovery'),
            'components':lambda:run_components(session, split='discovery'),
            'heads':lambda:run_heads(session, split='discovery', freeze=None),
        }
        for name, function in functions.items():
            torch.cuda.synchronize()
            before = time.monotonic()
            result = function()
            torch.cuda.synchronize()
            timings[name] = {'seconds':time.monotonic()-before, 'result':result,
                              'peak_allocated_gib':[torch.cuda.max_memory_allocated(i)/2**30 for i in range(2)]}
            write(args.output/'pilot_timings.json', timings)
            print(f'[pilot checkpoint] {name} complete', flush=True)
        identity.update(status='complete', timings_file='pilot_timings.json',
                        requires_resource_budget_review_before_science=True)
    except BaseException as error:
        identity.update(status='failed', error=repr(error))
        raise
    finally:
        write(args.output/'status.json', identity)


if __name__ == '__main__':
    main()
