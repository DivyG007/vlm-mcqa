"""Explicitly requested, read-only backup of cancelled Ada Qwen job45569."""
import hashlib,json,shutil,subprocess,tarfile,time,shlex,uuid
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'results/ada_completed_run_backups/job_45569'
HOST='astroduck@10.4.24.30'
PY='/home2/astroduck/envs/anlp-vlm-mcqa/hub_download_py39_v1/bin/python'

REMOTE=r'''
import gzip,hashlib,io,json,subprocess,sys,tarfile
from pathlib import Path
base=Path('/home2/astroduck/projects/anlp-vlm-mcqa/iconqa_20261028')
model=base.parent/'checkpoints/iconqa_qwen25vl_7b_cc594898137f460bfe9f0759e9844b3ce807cfb5'
files={}
for rel,label in [('results/full_checkpointed_wandb_75de070_attempt1','results'),('source_75de070','source'),('dataset','dataset')]:
 root=base/rel
 for f in sorted(root.rglob('*')):
  if f.is_symlink():
   target=f.resolve();wandb_logs=Path('/home2/astroduck/.cache/wandb/logs')
   if root.resolve() not in target.parents and not (wandb_logs in target.parents and target.suffix=='.log'):
    raise RuntimeError('external symlink '+str(f))
  if f.is_file():files[label+'/'+f.relative_to(root).as_posix()]=f
for name in ('full_45569.log','full_45569.err'):files['logs/'+name]=base/'logs'/name
files['provenance/submission.json']=base/'results/full_checkpointed_wandb_75de070_attempt1.submission.json'
files['provenance/source_75de070.tar.gz']=base/'source_75de070.tar.gz'
for f in sorted(model.iterdir()):
 if f.is_file() and f.suffix in ('.json','.txt','.jinja','.model'):files['model_metadata/'+f.name]=f
inventory={}
for name,f in files.items():
 h=hashlib.sha256()
 with f.open('rb') as stream:
  for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
 st=f.stat();inventory[name]={'bytes':st.st_size,'sha256':h.hexdigest(),'mtime_ns':st.st_mtime_ns,'remote_path':str(f)}
commands={'sacct':['sacct','-j','45569','--format=JobID,State,ExitCode,Elapsed,Start,End','-P'],
          'scontrol':['scontrol','show','job','45569','-o']}
accounting={k:subprocess.run(v,text=True,capture_output=True).__dict__ for k,v in commands.items()}
provenance={'job_id':'45569','run_state':'CANCELLED_BY_USER','scientific_status':'incomplete_confirmation',
 'manual_transfer_authorized':True,'remote_base':str(base),'source_commit':'75de070c9f0775e639dcc14333c83c0b3aa32767',
 'dataset_hash':'be51cf308b2e35ae','wandb_url':'https://wandb.ai/models-iiit-hyderabad5465/anlp-vlm-mcqa-staging/runs/a9515ee713ef',
 'accounting':accounting,'files':inventory}
raw=(json.dumps(provenance,indent=2,sort_keys=True)+'\n').encode()
print('Backing up %d files, %.1f MiB uncompressed'%(len(files),sum(v['bytes'] for v in inventory.values())/2**20),file=sys.stderr,flush=True)
with gzip.GzipFile(fileobj=sys.stdout.buffer,mode='wb',compresslevel=1) as compressed, tarfile.open(fileobj=compressed,mode='w|',dereference=True) as tar:
 info=tarfile.TarInfo('remote_inventory.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
 for index,(name,f) in enumerate(files.items(),1):
  st=f.stat();r=inventory[name]
  if st.st_size!=r['bytes'] or st.st_mtime_ns!=r['mtime_ns']:raise RuntimeError('remote file changed before copy '+name)
  tar.add(f,arcname=name,recursive=False)
  if index%100==0:print('Archived %d/%d files'%(index,len(files)),file=sys.stderr,flush=True)
'''


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def receive_archive(archive,err):
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15',HOST]
    export_name='job_45569_'+uuid.uuid4().hex[:12]
    producer=REMOTE.replace("with gzip.GzipFile(fileobj=sys.stdout.buffer,mode='wb',compresslevel=1) as compressed, tarfile.open(fileobj=compressed,mode='w|',dereference=True) as tar:",
        "export=base/'transfer_exports'/"+repr(export_name)+"\nexport.mkdir(parents=True,exist_ok=False)\narchive=export/'analysis_package.tar.gz'\nwith archive.open('xb') as bundle, gzip.GzipFile(fileobj=bundle,mode='wb',compresslevel=6) as compressed, tarfile.open(fileobj=compressed,mode='w|',dereference=True) as tar:")
    producer+='''\nh=hashlib.sha256()
with archive.open('rb') as f:
 for block in iter(lambda:f.read(8<<20),b''):h.update(block)
print(json.dumps({'path':str(archive),'bytes':archive.stat().st_size,'sha256':h.hexdigest()}),flush=True)
'''
    metadata=DEST/'transport_package.json'
    if metadata.exists():package=json.loads(metadata.read_text())
    else:
        with err.open('wb') as errors:
            result=subprocess.run(ssh+[PY,'-u','-'],input=producer.encode(),stdout=subprocess.PIPE,stderr=errors)
        if result.returncode:raise RuntimeError('remote archive preparation failed; see '+str(err))
        package=json.loads(result.stdout);metadata.write_text(json.dumps(package,indent=2)+'\n')
    print('Prepared temporary archive: %.1f MiB; downloading six parts'%(package['bytes']/2**20),flush=True)
    part_size=(package['bytes']+5)//6
    def download(index):
        start=index*part_size;length=min(part_size,package['bytes']-start)
        target=DEST/('.part_%02d'%index)
        for attempt in range(3):
            done=target.stat().st_size if target.exists() else 0
            if done==length:return target
            if done>length:raise ValueError('invalid partial segment size')
            program="from pathlib import Path;import sys\nf=Path(%r).open('rb');f.seek(%d);remaining=%d\nwhile remaining:\n b=f.read(min(1<<20,remaining))\n if not b:raise RuntimeError('short archive')\n sys.stdout.buffer.write(b);remaining-=len(b)\n"%(package['path'],start+done,length-done)
            with target.open('ab') as output,(DEST/('part_%02d.log'%index)).open('ab') as errors:
                child=subprocess.Popen(ssh+['-n',PY+' -c '+shlex.quote(program)],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=errors)
                while True:
                    block=child.stdout.read(1<<20)
                    if not block:break
                    output.write(block);output.flush()
                code=child.wait()
            if code==0 and target.stat().st_size==length:
                print('Received archive part %d/6'%(index+1),flush=True);return target
        raise RuntimeError('archive segment transfer failed '+str(index))
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(download,i) for i in range(6)]
        for future in as_completed(futures):future.result()
    with archive.open('wb') as output:
        for i in range(6):
            with (DEST/('.part_%02d'%i)).open('rb') as source:shutil.copyfileobj(source,output,8<<20)
    if archive.stat().st_size!=package['bytes'] or digest(archive)!=package['sha256']:raise ValueError('transport archive hash mismatch')
    return package,ssh


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    receipt=DEST/'transfer_complete.json'
    if receipt.exists():
        saved=json.loads(receipt.read_text());manifest=json.loads((DEST/'remote_inventory.json').read_text())
        if digest(DEST/'remote_inventory.json')!=saved['inventory_sha256']:raise ValueError('local inventory changed')
        if all((DEST/n).is_file() and digest(DEST/n)==r['sha256'] for n,r in manifest['files'].items()):
            print('Already transferred and verified: '+str(DEST));return
        raise ValueError('existing verified backup changed; preserve and inspect before repair')
    if shutil.disk_usage(DEST).free<2*2**30:raise RuntimeError('at least2GiB free space required')
    archive=DEST/'.incoming.tar.gz';err=DEST/'transfer_stderr.log'
    started=time.time()
    package,ssh=receive_archive(archive,err)
    # Extract only regular files/directories, inside this dedicated backup folder.
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            rel=PurePosixPath(member.name)
            if rel.is_absolute() or '..' in rel.parts or '\\' in member.name or ':' in member.name:
                raise ValueError('unsafe archive member')
            target=DEST.joinpath(*rel.parts).resolve()
            if DEST.resolve() not in target.parents:raise ValueError('archive escapes backup directory')
            if member.isdir():target.mkdir(parents=True,exist_ok=True);continue
            if not member.isfile():raise ValueError('nonregular archive member')
            target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as source,target.open('wb') as output:shutil.copyfileobj(source,output,8<<20)
    manifest=json.loads((DEST/'remote_inventory.json').read_text())
    for name,r in manifest['files'].items():
        f=DEST/name
        if f.stat().st_size!=r['bytes'] or digest(f)!=r['sha256']:raise ValueError('transfer hash mismatch '+name)
    readme='''# Qwen batch-one job45569 — local analysis package

This run was intentionally cancelled; confirmation is incomplete. This is a
verified backup, not a successful-run or scientific-validity declaration.

- results/: all saved predictions, causal records, plots, summaries, checkpoints
  and local W&B files; discovery completed, confirmation was interrupted.
- dataset/: exact run dataset, all manifests and image assets.
- source/: immutable executed source snapshot, configs and source manifest.
- logs/: stdout and stderr. provenance/: submission, original source archive.
- model_metadata/: checkpoint verification, architecture/processor/tokenizer
  metadata; model weight shards and environment binaries are excluded.
- remote_inventory.json: per-file SHA256, bytes, original Ada paths, final
  accounting, source/model provenance and W&B URL.
- transfer_complete.json: verified local-copy receipt.

Original absolute Ada paths inside JSON artifacts are provenance; corresponding
local files are mapped by remote_inventory.json. Do not interpret incomplete
confirmation records as finished experiments. Remote files were not changed.
'''
    (DEST/'README.md').write_text(readme,encoding='utf-8')
    saved={'job_id':'45569','transfer_status':'verified','transfer_complete':True,
           'run_state':'CANCELLED_BY_USER','authorization':'explicit user request to transfer cancelled batch1',
           'destination':str(DEST),'source_commit':manifest['source_commit'],
           'remote_result_path':manifest['remote_base']+'/results/full_checkpointed_wandb_75de070_attempt1',
           'files':len(manifest['files']),'bytes':sum(r['bytes'] for r in manifest['files'].values()),
           'inventory_sha256':digest(DEST/'remote_inventory.json'),
           'completed_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':round(time.time()-started,2),
           'transport':'temporary Ada archive downloaded in six parallel segments','transport_sha256':package['sha256']}
    pending=DEST/'transfer_complete.pending';pending.write_text(json.dumps(saved,indent=2)+'\n');pending.replace(receipt)
    archive.unlink()  # Only our verified temporary transport archive, not results.
    for i in range(6):(DEST/('.part_%02d'%i)).unlink()
    cleanup="from pathlib import Path\np=Path(%r);root=Path('/home2/astroduck/projects/anlp-vlm-mcqa/iconqa_20261028/transfer_exports').resolve()\nassert root in p.resolve().parents and p.name=='analysis_package.tar.gz'\np.unlink();p.parent.rmdir()\n"%package['path']
    cleaned=subprocess.run(ssh+[PY+' -c '+shlex.quote(cleanup)],capture_output=True)
    if cleaned.returncode:print('Verified locally; temporary Ada archive cleanup needs retry: '+package['path'],flush=True)
    print(json.dumps(saved,indent=2))


if __name__=='__main__':main()
