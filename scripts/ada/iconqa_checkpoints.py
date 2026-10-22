"""Hash-bound method checkpoints. Partial methods are preserved and rerun."""
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tuples(value):
    return tuple(tuples(x) for x in value) if isinstance(value,list) else value


class Blocks:
    def __init__(self, root, identity, rng):
        self.root=Path(root); self.root.mkdir(exist_ok=True,parents=True)
        self.identity=identity; self.rng=rng

    def run(self,name,action,artifacts):
        marker=self.root/(name+'.json')
        if marker.exists():
            saved=json.loads(marker.read_text())
            if saved['identity']!=self.identity:
                raise ValueError('checkpoint identity mismatch: '+name)
            for path,expected in saved['artifacts'].items():
                if not Path(path).is_file() or sha(path)!=expected:
                    raise ValueError('checkpoint artifact changed: '+path)
            self.rng.setstate(tuples(saved['rng_after']))
            print('[resume] verified '+name,flush=True)
            return saved['result']
        existing=[p for p in artifacts() if p.is_file()]
        if existing:
            archive=self.root/('partial_'+name+'_'+str(time.time_ns()));archive.mkdir()
            for path in existing:
                path.rename(archive/path.name)
        result=action()
        files=[p for p in artifacts() if p.is_file()]
        if not files:
            raise ValueError('no durable method artifacts: '+name)
        record={'identity':self.identity,'result':result,'rng_after':self.rng.getstate(),
                'artifacts':{str(p.resolve()):sha(p) for p in files}}
        temporary=marker.with_suffix('.pending')
        temporary.write_text(json.dumps(record,sort_keys=True,default=str,allow_nan=False)+'\n')
        temporary.replace(marker)
        return result
