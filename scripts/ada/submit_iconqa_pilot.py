"""Run on Ada gateway: submit once with actual-node exclusions and a receipt."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'dataset-root', 'checkpoint-root', 'output-root', 'log-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--account', default='cvit')
    p.add_argument('--stage', choices=('pilot','full','batch','small'), default='pilot')
    p.add_argument('--hours',type=int,default=48)
    p.add_argument('--batch-size',type=int,choices=(1,2,4),default=1)
    p.add_argument('--batch-evidence',type=Path)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--test-only',action='store_true')
    p.add_argument('--calibration-only',action='store_true')
    args = p.parse_args()
    if args.resume and args.stage not in ('full','small'):
        p.error('--resume is implemented for full campaigns only')
    if args.stage=='small' and args.batch_size!=4:
        p.error('small Qwen campaign is configured for --batch-size 4')
    if args.calibration_only and args.stage!='small':p.error('--calibration-only requires --stage small')
    if args.stage!='small' and args.batch_size!=1 and (args.stage!='full' or not args.batch_evidence or not args.batch_evidence.is_file()):
        p.error('full batch 2/4 requires an existing --batch-evidence report')
    source, dataset, checkpoint, output, logs = [getattr(args,n).resolve() for n in
        ('source_root','dataset_root','checkpoint_root','output_root','log_root')]
    receipt = output.with_name(output.name+'.submission.json')
    if args.resume:
        receipt=output.with_name(output.name+'.resume_'+str(time.time_ns())+'.submission.json')
    if receipt.exists() or (output.exists()!=args.resume):
        raise ValueError('fresh result and submission receipt paths required')
    if not (checkpoint/'verified_checkpoint.json').is_file():
        raise ValueError('publisher-verified checkpoint required before reserving GPUs')
    names = set(subprocess.check_output(['sinfo','-N','-p','u22','-h','-o','%N'], text=True).split())
    eligible = {n for n in names if n.startswith('gnode') and n[5:].isdigit() and int(n[5:]) > 50
                and n not in {'gnode065','gnode066','gnode077'}}
    if not eligible:
        raise ValueError('no suitable Ada nodes advertised')
    excluded = sorted(names-eligible)
    if any(',' in str(path) for path in (source,dataset,checkpoint,output)):
        raise ValueError('Slurm --export paths cannot contain commas')
    script = source/'scripts/ada/iconqa_qwen_pilot.sbatch'
    if b'\r' in script.read_bytes():
        raise ValueError('Slurm script must have Unix line endings')
    output.parent.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    if args.stage=='small':
        import sys
        subprocess.run([sys.executable,str(source/'scripts/ada/iconqa_qwen_small.py'),'--preflight',
                        '--source-root',str(source),'--dataset',str(dataset),'--checkpoint',str(checkpoint),
                        '--output',str(output),'--batch-size','4'],check=True)
    command = ['sbatch','--parsable','--account='+args.account,'--exclude='+','.join(excluded),
               '--job-name=iconqa_qwen_'+args.stage, '--time='+(str(args.hours)+':00:00' if args.stage in ('full','small') else '04:00:00'),
               '--output='+str(logs/(args.stage+'_%j.log')),'--error='+str(logs/(args.stage+'_%j.err')),
               '--export=ALL,SOURCE_ROOT='+str(source)+',DATASET_ROOT='+str(dataset)+
               ',CHECKPOINT_ROOT='+str(checkpoint)+',OUTPUT_ROOT='+str(output)+
               ',CAMPAIGN_ENTRY='+({'full':'iconqa_qwen_campaign.py','small':'iconqa_qwen_small.py'}.get(args.stage,'iconqa_qwen_pilot.py'))+
               ',ICONQA_RESUME='+('1' if args.resume else '0'),str(script)]
    if args.stage=='batch':
        command[-2]+=',ICONQA_BATCH_BENCHMARK=1'
    else:
        command[-2]+=',ICONQA_BATCH_BENCHMARK=0'
    command[-2]+=',ICONQA_BATCH_SIZE='+str(args.batch_size)
    command[-2]+=',ICONQA_CALIBRATION_ONLY='+('1' if args.calibration_only else '0')
    if args.batch_evidence:
        evidence=str(args.batch_evidence.resolve())
        if ',' in evidence:
            p.error('batch evidence path cannot contain commas')
        command[-2]+=',ICONQA_BATCH_EVIDENCE='+evidence
    if args.test_only:
        subprocess.run(command[:1]+['--test-only']+command[1:],check=True)
        return
    reply = subprocess.check_output(command,text=True).strip()
    job = reply.split(';')[0]
    if not job.isdigit():
        raise ValueError('unexpected sbatch reply: '+reply)
    with receipt.open('x') as stream:
        json.dump({'job_id':job, 'command':command, 'source_root':str(source),
                   'output_root':str(output), 'source_snapshot_sha256':hashlib.sha256((source/'source_snapshot.json').read_bytes()).hexdigest(),
                   'stage':args.stage, 'status':'submitted_not_yet_verified'},stream,indent=2)
    print(job)


if __name__ == '__main__':
    main()
