"""Completed-only IconQA backup, with per-run receipts and verified shared assets."""
import argparse,hashlib,json,shutil,subprocess,tarfile,shlex
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath
import backup_iconqa_batch1 as transport

ROOT=Path(__file__).resolve().parents[2]
BACKUPS=ROOT/'results/ada_completed_run_backups'
OLD='/home2/astroduck/projects/anlp-vlm-mcqa/iconqa_20261028'
SMALL='/home2/astroduck/projects/anlp-vlm-mcqa/iconqa_small_20261029'
SPECS={
 '45585':dict(base=OLD,result='full_batch4_fedffdd_attempt1',source='fedffdd',log='full',dataset='dataset',pilot='batch_1351753_attempt1',small=False,layers=28),
 '46804':dict(base=SMALL,result='qwen_small_b4_attempt1',source='76939be',log='small',dataset='dataset_v1',pilot=None,small=True,layers=28),
 '46912':dict(base=SMALL,result='internvl_small_b4_attempt1',source='724f9ca',log='small_internvl',dataset='dataset_v1',pilot='internvl35_8b_pilot_attempt2',small=True,layers=36),
 '46918':dict(base=SMALL,result='llava_small_b2_attempt1',source='50946d0',log='small_llava',dataset='dataset_v1',pilot='llava_ov_7b_si_pilot_attempt3',small=True,layers=28)}

REMOTE=r'''
import gzip,hashlib,io,json,subprocess,sys,tarfile
from pathlib import Path
cfg=CONFIG
base=Path(cfg['base']);job=cfg['job'];result=base/'results'/cfg['result'];files={}
accounting=subprocess.check_output(['sacct','-X','-n','-j',job,'--format=JobID,State,ExitCode','-P'],text=True).strip()
if not any(line.split('|')[:3]==[job,'COMPLETED','0:0'] for line in accounting.splitlines()):raise RuntimeError('job not COMPLETED/0:0')
state=json.loads((result/'campaign_status.json').read_text())
if state['status']!='execution_finished_pending_independent_audit':raise RuntimeError('campaign not finished')
report=json.loads((result/'confirmation/confirmation_report.json').read_text())
if not report['checks']['execution_complete']:raise RuntimeError('confirmation execution incomplete')
raw_counts={}
methods=['behavior','lens','patching','components','heads','clevr_transfer']+(['edit_controls'] if cfg['small'] else [])
for phase in ('discovery','confirmation'):
 p=result/phase
 for name in methods:
  if not (p/'.checkpoints'/(name+'.json')).is_file():raise RuntimeError('missing checkpoint '+phase+'/'+name)
 if not json.loads((p/'calibration.json').read_text())['all_passed']:raise RuntimeError('calibration failed')
 if json.loads((p/'eligibility.json').read_text())['pairs_used']!=(32 if cfg['small'] else 64):raise RuntimeError('pair coverage')
 expect={'patching':(576*cfg['layers']+256*4*(cfg['layers']//2+1)) if cfg['small'] else 151040,
         'components':384*2*cfg['layers'] if cfg['small'] else 114688,
         'component_ablation':32*2*cfg['layers'] if cfg['small'] else 7168}
 for stem,count in expect.items():
  f=p/(stem+'_'+phase+'.jsonl')
  with f.open() as stream:n=sum(1 for line in stream if line.strip())
  if n!=count:raise RuntimeError('raw count mismatch '+str(f)+' '+str(n))
  raw_counts[phase+'/'+stem]=n
roots=[(result,'results'),(base/('source_'+cfg['source']),'source'),(base/cfg['dataset'],'dataset')]
if cfg['pilot']:roots.append((base/'results'/cfg['pilot'],'pilot_evidence'))
for root,label in roots:
 for f in sorted(root.rglob('*')):
  if f.is_symlink():
   target=f.resolve()
   if root.resolve() not in target.parents and not (Path('/home2/astroduck/.cache/wandb/logs') in target.parents and target.suffix=='.log'):raise RuntimeError('external symlink')
  if f.is_file():files[label+'/'+f.relative_to(root).as_posix()]=f
for suffix in ('log','err'):files['logs/'+cfg['log']+'_'+job+'.'+suffix]=base/'logs'/(cfg['log']+'_'+job+'.'+suffix)
for f in sorted((base/'results').glob(cfg['result']+'*.submission*.json')):files['provenance/'+f.name]=f
source=json.loads((base/('source_'+cfg['source'])/'source_snapshot.json').read_text())
model=Path('/home2/astroduck/projects/anlp-vlm-mcqa/checkpoints/iconqa_qwen25vl_7b_cc594898137f460bfe9f0759e9844b3ce807cfb5')
if job=='46912':model=Path('/share1/astroduck/anlp-vlm-mcqa/checkpoints/iconqa_internvl35_8b_741a7d03')
if job=='46918':model=Path('/home2/astroduck/projects/anlp-vlm-mcqa/checkpoints/iconqa_llava_ov_7b_si_cae7503d')
for f in sorted(model.iterdir()):
 if f.is_file() and f.suffix in ('.json','.txt','.jinja','.model'):files['model_metadata/'+f.name]=f
inventory={};payload={}
for name,f in files.items():
 h=hashlib.sha256()
 with f.open('rb') as stream:
  for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
 st=f.stat();r={'bytes':st.st_size,'sha256':h.hexdigest(),'mtime_ns':st.st_mtime_ns,'remote_path':str(f)}
 shared=cfg['reuse'].get(str(f))
 if shared and shared['sha256']==r['sha256']:r['local_path']=shared['local_path']
 else:payload[name]=f
 inventory[name]=r
for phase in ('discovery','confirmation'):
 for marker in (result/phase/'.checkpoints').glob('*.json'):
  for name,sha in json.loads(marker.read_text())['artifacts'].items():
   if not any(r['remote_path']==name and r['sha256']==sha for r in inventory.values()):raise RuntimeError('checkpoint artifact mismatch '+name)
provenance={'job_id':job,'run_state':'COMPLETED','scientific_status':'pending_independent_audit','source_commit':source['source_git_commit'],
 'remote_base':str(base),'remote_result_path':str(result),'files':inventory,'raw_counts':raw_counts,'campaign_status':state,
 'sacct':subprocess.check_output(['sacct','-j',job,'--format=JobID,State,ExitCode,Elapsed,Start,End','-P'],text=True)}
raw=(json.dumps(provenance,indent=2,sort_keys=True)+'\n').encode()
print('Verified completion/counts; transferring %d files; reusing %d assets'%(len(payload),len(files)-len(payload)),file=sys.stderr,flush=True)
with gzip.GzipFile(fileobj=sys.stdout.buffer,mode='wb',compresslevel=1) as compressed, tarfile.open(fileobj=compressed,mode='w|',dereference=True) as tar:
 info=tarfile.TarInfo('remote_inventory.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
 for index,(name,f) in enumerate(payload.items(),1):
  r=inventory[name];st=f.stat()
  if st.st_size!=r['bytes'] or st.st_mtime_ns!=r['mtime_ns']:raise RuntimeError('file changed')
  tar.add(f,arcname=name,recursive=False)
  if index%100==0:print('Archived '+str(index),file=sys.stderr,flush=True)
'''


def path_for(dest,name,r):return Path(r['local_path']) if 'local_path' in r else dest/name


def main():
 p=argparse.ArgumentParser();p.add_argument('--job',choices=SPECS,required=True);a=p.parse_args();cfg={**SPECS[a.job],'job':a.job,'reuse':{}}
 dest=BACKUPS/('job_'+a.job);dest.mkdir(parents=True,exist_ok=True);receipt=dest/'transfer_complete.json'
 if receipt.exists():
  saved=json.loads(receipt.read_text());manifest=json.loads((dest/'remote_inventory.json').read_text())
  assert saved['job_id']==a.job and saved['inventory_sha256']==transport.digest(dest/'remote_inventory.json')
  assert saved['source_commit'].startswith(cfg['source']) and saved['remote_result_path']==cfg['base']+'/results/'+cfg['result']
  assert all(path_for(dest,n,r).is_file() and transport.digest(path_for(dest,n,r))==r['sha256'] for n,r in manifest['files'].items())
  print('Already transferred and verified '+a.job);return
 # Only verified assets are offered for reuse; result payloads are always per-run.
 old=BACKUPS/'job_45569'
 if (old/'transfer_complete.json').exists():
  m=json.loads((old/'remote_inventory.json').read_text())
  for n,r in m['files'].items():
   if n.startswith(('dataset/','model_metadata/')) and (old/n).is_file() and transport.digest(old/n)==r['sha256']:
    cfg['reuse'][r['remote_path']]={'local_path':str(old/n),'sha256':r['sha256']}
 if cfg['small']:
  local=ROOT/'data/generated/iconqa_cf_core7_small_v1_20261029_attempt2'
  for f in local.rglob('*'):
   if f.is_file():cfg['reuse'][cfg['base']+'/'+cfg['dataset']+'/'+f.relative_to(local).as_posix()]={'local_path':str(f),'sha256':transport.digest(f)}
 if shutil.disk_usage(dest).free<3*2**30:raise RuntimeError('insufficient disk space')
 transport.DEST=dest;transport.REMOTE=REMOTE.replace('cfg=CONFIG','cfg='+repr(cfg))
 archive=dest/'.incoming.tar.gz';package,ssh=transport.receive_archive(archive,dest/'transfer_stderr.log')
 with tarfile.open(archive,'r:gz') as tar:
  for member in tar:
   rel=PurePosixPath(member.name);target=dest.joinpath(*rel.parts).resolve()
   if rel.is_absolute() or '..' in rel.parts or ':' in member.name or '\\' in member.name or dest.resolve() not in target.parents or not member.isfile():raise ValueError('unsafe archive member')
   target.parent.mkdir(parents=True,exist_ok=True)
   with tar.extractfile(member) as source,target.open('wb') as output:shutil.copyfileobj(source,output,8<<20)
 manifest=json.loads((dest/'remote_inventory.json').read_text())
 for n,r in manifest['files'].items():
  f=path_for(dest,n,r)
  if f.stat().st_size!=r['bytes'] or transport.digest(f)!=r['sha256']:raise ValueError('file verification failed '+n)
 shared={n:r['local_path'] for n,r in manifest['files'].items() if 'local_path' in r}
 (dest/'shared_assets.json').write_text(json.dumps(shared,indent=2)+'\n')
 (dest/'README.md').write_text('# Completed IconQA job '+a.job+'\n\nExecution completed; independent scientific audit remains pending.\nResults/logs/source/provenance are local. Reused dataset/model assets are mapped\nin shared_assets.json and remote_inventory.json; retain those referenced folders.\nNo model weights or credentials are included.\n')
 saved={'job_id':a.job,'transfer_status':'verified','transfer_complete':True,'run_state':'COMPLETED','scientific_audit':'pending',
  'destination':str(dest),'source_commit':manifest['source_commit'],'remote_result_path':manifest['remote_result_path'],
  'files':len(manifest['files']),'transferred_files':len(manifest['files'])-len(shared),'reused_files':len(shared),
  'bytes':sum(r['bytes'] for r in manifest['files'].values()),'downloaded_file_bytes':sum(r['bytes'] for r in manifest['files'].values() if 'local_path' not in r),
  'inventory_sha256':transport.digest(dest/'remote_inventory.json'),'completed_utc':datetime.now(timezone.utc).isoformat()}
 pending=dest/'transfer_complete.pending';pending.write_text(json.dumps(saved,indent=2)+'\n');pending.replace(receipt)
 archive.unlink()
 for i in range(6):(dest/('.part_%02d'%i)).unlink()
 cleanup="from pathlib import Path\np=Path(%r);root=Path(%r).resolve()\nassert root in p.resolve().parents and p.name=='analysis_package.tar.gz'\np.unlink();p.parent.rmdir()"%(package['path'],cfg['base']+'/transfer_exports')
 subprocess.run(ssh+['-n',transport.PY+' -c '+shlex.quote(cleanup)],stdin=subprocess.DEVNULL,check=True)
 print(json.dumps(saved,indent=2),flush=True)


if __name__=='__main__':main()
