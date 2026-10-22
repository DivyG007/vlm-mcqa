"""Gateway-only submission of pinned model-specific batch pilots."""
import argparse,json,subprocess,sys,shlex,time
from pathlib import Path
from iconqa_small_inputs import verify_release,digest
from stage_iconqa_transfer import MODELS


def main():
    p=argparse.ArgumentParser()
    for k in ('source-root','dataset-root','checkpoint-root','output-root','log-root'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--model',choices=MODELS,required=True);p.add_argument('--afterany',type=int);p.add_argument('--test-only',action='store_true')
    p.add_argument('--stage-local',action='store_true',help='Copy gateway-only weights into allocated-node scratch before loading')
    args=p.parse_args();source,dataset,checkpoint,out,logs=[getattr(args,k).resolve() for k in ('source_root','dataset_root','checkpoint_root','output_root','log_root')]
    receipt=out.with_name(out.name+'.submission.json')
    if out.exists() or receipt.exists():raise ValueError('fresh output and submission required')
    verify_release(dataset)
    ck=json.loads((checkpoint/'verified_checkpoint.json').read_text());model,revision=MODELS[args.model]
    if (ck.get('model_id'),ck.get('revision'),ck.get('status'))!=(model,revision,'verified'):raise ValueError('model not publisher-verified')
    snapshot=json.loads((source/'source_snapshot.json').read_text())
    for name,sha in snapshot['files'].items():
        if digest(source/name)!=sha:raise ValueError('source changed')
    script=source/'scripts/ada/iconqa_transfer_pilot.sbatch'
    if b'\r' in script.read_bytes():raise ValueError('Unix script required')
    if any(',' in str(p) for p in (source,dataset,checkpoint,out)):raise ValueError('unsupported comma in path')
    logs.mkdir(exist_ok=True,parents=True);out.parent.mkdir(exist_ok=True,parents=True)
    runtime_checkpoint=Path('/scratch/astroduck/iconqa_transfer')/out.name/'checkpoint' if args.stage_local else checkpoint
    command=['sbatch','--parsable','--job-name=iconqa_'+args.model+'_pilot',
             '--output='+str(logs/('transfer_'+args.model+'_%j.log')),
             '--error='+str(logs/('transfer_'+args.model+'_%j.err')),
             '--export=ALL,SOURCE_ROOT='+str(source)+',DATASET_ROOT='+str(dataset)+',CHECKPOINT_ROOT='+str(runtime_checkpoint)+
             ',OUTPUT_ROOT='+str(out)+',PILOT_MODEL='+args.model+',WAIT_FOR_STAGING='+str(int(args.stage_local))]
    if args.afterany:command.append('--dependency=afterany:'+str(args.afterany))
    command.append(str(script))
    if args.test_only:subprocess.run(command[:1]+['--test-only']+command[1:],check=True);return
    reply=subprocess.check_output(command,text=True).strip();job=reply.split(';')[0]
    if not job.isdigit():raise ValueError('unexpected job ID')
    with receipt.open('x') as f:json.dump({'job_id':job,'model':args.model,'command':command,'source_manifest':digest(source/'source_snapshot.json'),
                                        'output_root':str(out),'status':'submitted_not_verified'},f,indent=2)
    print(job,flush=True)
    if args.stage_local:
        from iconqa_node_staging import stage_checkpoint
        stage_checkpoint(job,checkpoint,runtime_checkpoint)


if __name__=='__main__':main()
