"""Gateway-to-allocated-node verified checkpoint stream; no GPU model load."""
import subprocess,time


def stage_checkpoint(job,checkpoint,runtime_checkpoint):
    while True:
        state=subprocess.check_output(['squeue','-h','-j',job,'-o','%T'],text=True).strip()
        if state=='RUNNING':break
        if not state or state not in ('PENDING','CONFIGURING'):raise RuntimeError('job cannot receive checkpoint: '+state)
        time.sleep(10)
    step=['srun','--jobid='+job,'--overlap','--nodes=1','--ntasks=1']
    try:
        # One gateway-to-compute stream; no checkpoint bytes pass through the laptop.
        subprocess.run(step+['mkdir','-p',str(runtime_checkpoint)],check=True)
        producer=subprocess.Popen(['tar','--exclude=.cache','-C',str(checkpoint),'-cf','-','.'],stdout=subprocess.PIPE)
        consumer=subprocess.run(step+['tar','-C',str(runtime_checkpoint),'-xf','-'],stdin=producer.stdout)
        producer.stdout.close()
        if producer.wait()!=0 or consumer.returncode!=0:raise RuntimeError('checkpoint transfer failed')
        verifier='''import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]);receipt=json.loads((p/'verified_checkpoint.json').read_text())
for name,r in receipt['files'].items():
 h=hashlib.sha256()
 with (p/name).open('rb') as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 if h.hexdigest()!=r['sha256']:raise RuntimeError('staged hash mismatch '+name)
(p/'LOCAL_STAGE_READY').write_text('all publisher-verified file hashes match\\n')
'''
        subprocess.run(step+['/usr/bin/python3','-c',verifier,str(runtime_checkpoint)],check=True)
        print('Node-local checkpoint verified for job '+job,flush=True)
    except BaseException:
        subprocess.run(step+['touch',str(runtime_checkpoint/'LOCAL_STAGE_FAILED')],check=False)
        raise
