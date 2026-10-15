"""Export compact LLaVA evidence without raw records or W&B working caches."""
import argparse
import json
from pathlib import Path
import shutil

from audit_llava_completed import read, sha


def sanitized(value):
    if isinstance(value, dict):
        return {k: sanitized(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitized(v) for v in value]
    if isinstance(value, str):
        return value.replace('/scratch/yashav/VLMCQA', '<pkgpu-project-root>')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    audit = read(args.audit_root / 'local_audit.json')
    publication = read(args.audit_root / 'llava_wandb_audit_20261019.json')
    assert audit['passed'] and publication['verified']
    campaign = args.audit_root / '20261016_124107'
    args.out.mkdir(parents=True, exist_ok=True)
    entries = {}

    def export(source, relative):
        target = args.out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == '.json':
            target.write_text(json.dumps(sanitized(read(source)), indent=2) + '\n', encoding='utf-8', newline='\n')
        elif source.suffix == '.csv':
            target.write_bytes(source.read_bytes().replace(b'\r\n', b'\n'))
        else:
            shutil.copyfile(source, target)
        entries[str(relative).replace('\\', '/')] = {'original_sha256': sha(source),
                                                    'export_sha256': sha(target)}

    export(campaign / 'campaign.json', Path('campaign.json'))
    export(campaign / 'models.json', Path('models.json'))
    export(campaign / 'preflight.json', Path('runtime_preflight.json'))
    export(campaign / 'llava_ov_7b_si_checkpoint.json', Path('checkpoint_manifest.json'))
    export(campaign / 'llava_core7/freeze.json', Path('freeze.json'))
    for split in ('discovery', 'confirmation'):
        folder = campaign / 'llava_core7' / split
        for source in folder.glob('*.json'):
            if not source.name.startswith('wandb_'):
                export(source, Path(split) / source.name)
        for pattern in ('tables/*.csv', 'figures/*.png'):
            for source in folder.glob(pattern):
                export(source, Path(split) / source.relative_to(folder))
    export(args.audit_root / 'local_audit.json', Path('audit/local_audit.json'))
    export(args.audit_root / 'llava_wandb_audit_20261019.json', Path('audit/wandb_publication.json'))
    manifest = {'exported_on': '2026-10-19', 'campaign': '20261016_124107',
                'execution_source_commit': audit['campaign']['source_git_commit'],
                'archive_sha256': '607d076d52ab244d50503f6f0f1e1371c80eb6efabfa0b2b952684096d800d00',
                'archive_bytes': 33745556, 'paths_sanitized': True,
                'note': 'Stored freeze_hash identifies the original record; machine paths are sanitized in this export.',
                'files': entries}
    (args.out / 'export_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8', newline='\n')
    print('EXPORTED', len(entries), 'files; raw JSONL/logs/caches remain in the ignored evidence mirror.')


if __name__ == '__main__':
    main()
