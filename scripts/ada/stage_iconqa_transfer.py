"""Pinned anonymous acquisition of the historical LLaVA/InternVL checkpoints."""
import argparse,hashlib,json
from pathlib import Path

MODELS={'llava_ov_7b_si':('llava-hf/llava-onevision-qwen2-7b-si-hf','cae7503dce88887557f73db1882328d21d9a7e21'),
        'internvl35_8b':('OpenGVLab/InternVL3_5-8B-HF','741a7d03020411e666c6109218ab71e08151ef86')}


def main():
    from huggingface_hub import HfApi,snapshot_download
    p=argparse.ArgumentParser();p.add_argument('--model',choices=MODELS,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();model,revision=MODELS[args.model];root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/'verified_checkpoint.json').exists():raise ValueError('verified model already staged; inspect/reuse')
    print('Acquiring '+model+' revision '+revision,flush=True)
    info=HfApi(token=False).model_info(model,revision=revision,files_metadata=True)
    if info.sha!=revision:raise ValueError('publisher revision changed')
    snapshot_download(model,revision=revision,local_dir=root,token=False,
                      allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','*.model'],max_workers=2)
    inventory={}
    for item in info.siblings:
        path=root/item.rfilename
        if not path.is_file():continue
        sha=hashlib.sha256();blob=hashlib.sha1(f'blob {path.stat().st_size}\0'.encode())
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8<<20),b''):sha.update(chunk);blob.update(chunk)
        lfs=getattr(item,'lfs',None);expected=getattr(lfs,'sha256',None) if lfs else None
        if expected and sha.hexdigest()!=expected:raise ValueError('LFS hash mismatch '+item.rfilename)
        if not expected and item.blob_id and blob.hexdigest()!=item.blob_id:raise ValueError('Git blob mismatch '+item.rfilename)
        inventory[item.rfilename]={'sha256':sha.hexdigest(),'bytes':path.stat().st_size,
                                   'publisher_lfs_sha256':expected,'publisher_blob_id':item.blob_id}
    index=json.loads((root/'model.safetensors.index.json').read_text())
    if not set(index['weight_map'].values())<=inventory.keys():raise ValueError('weight shard missing')
    receipt={'model_key':args.model,'model_id':model,'revision':revision,'status':'verified','files':inventory,
             'staging_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with (root/'verified_checkpoint.json').open('x') as f:json.dump(receipt,f,indent=2,sort_keys=True)
    print(json.dumps({'status':'verified','model_key':args.model,'bytes':sum(r['bytes'] for r in inventory.values())}),flush=True)


if __name__=='__main__':main()
