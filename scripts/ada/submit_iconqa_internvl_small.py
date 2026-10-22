"""Prepare/validate InternVL small campaign; submission requires explicit --submit."""
import argparse,json,subprocess,sys,uuid
from pathlib import Path
from iconqa_small_inputs import digest,verify_release
from iconqa_internvl_contract import verify_checkpoint,verify_pilot


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('source-root','dataset-root','checkpoint-root','pilot-evidence','output-root','log-root'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--submit',action='store_true');p.add_argument('--resume',action='store_true')
    args=p.parse_args();source,dataset,checkpoint,pilot,out,logs=[getattr(args,k).resolve() for k in
        ('source_root','dataset_root','checkpoint_root','pilot_evidence','output_root','log_root')]
    if any(',' in str(f) for f in (source,dataset,checkpoint,pilot,out)):raise ValueError('unsupported comma in path')
    if out.exists()!=args.resume:raise ValueError('fresh output required, or explicit --resume for existing output')
    if args.resume and not (out/'resume_identity.json').is_file():raise ValueError('missing resume identity')
    verify_release(dataset);verify_checkpoint(checkpoint);evidence=verify_pilot(pilot)
    manifest=json.loads((source/'source_snapshot.json').read_text())
    for name,sha in manifest['files'].items():
        if digest(source/name)!=sha:raise ValueError('changed source '+name)
    if args.resume:
        old=json.loads((out/'resume_identity.json').read_text())
        if old['source_manifest']!=digest(source/'source_snapshot.json') or old['pilot_evidence']!=evidence['sha256']:
            raise ValueError('resume source or pilot changed')
    script=source/'scripts/ada/iconqa_internvl_small.sbatch'
    if b'\r' in script.read_bytes():raise ValueError('Unix batch script required')
    subprocess.run([sys.executable,str(source/'scripts/ada/iconqa_internvl_small.py'),
        '--source-root',str(source),'--dataset',str(dataset),'--checkpoint',str(checkpoint),
        '--pilot-evidence',str(pilot),'--output',str(out),'--preflight'],check=True)
    logs.mkdir(parents=True,exist_ok=True);out.parent.mkdir(parents=True,exist_ok=True)
    # A fresh cache path per submission prevents stale READY markers on a resumed node.
    runtime_checkpoint=Path('/scratch/astroduck/iconqa_transfer')/(out.name+'_'+uuid.uuid4().hex[:8])/'checkpoint'
    command=['sbatch','--parsable','--output='+str(logs/'small_internvl_%j.log'),
             '--error='+str(logs/'small_internvl_%j.err'),
             '--export=ALL,SOURCE_ROOT='+str(source)+',DATASET_ROOT='+str(dataset)+',CHECKPOINT_ROOT='+str(runtime_checkpoint)+
             ',PILOT_EVIDENCE='+str(pilot)+',OUTPUT_ROOT='+str(out)+',RESUME='+str(int(args.resume))+',WAIT_FOR_STAGING=1',str(script)]
    if not args.submit:
        subprocess.run(command[:1]+['--test-only']+command[1:],check=True)
        print(json.dumps({'status':'prepared_not_submitted','source_commit':manifest['source_git_commit'],
                          'batch_size':4,'gpus':2,'limit_hours':48,'output':str(out),'command':command}));return
    reply=subprocess.check_output(command,text=True).strip();job=reply.split(';')[0]
    if not job.isdigit():raise ValueError('unexpected job ID')
    receipt=out.with_name(out.name+'.submission_'+job+'.json')
    with receipt.open('x') as f:json.dump({'job_id':job,'command':command,'source_manifest':digest(source/'source_snapshot.json'),
                                        'pilot_evidence':evidence['sha256'],'resume':args.resume},f,indent=2)
    print(job,flush=True)
    from iconqa_node_staging import stage_checkpoint
    stage_checkpoint(job,checkpoint,runtime_checkpoint)


if __name__=='__main__':main()
