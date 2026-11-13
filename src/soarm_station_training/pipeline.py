import hashlib
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
import yaml

def audit(root):
    root=Path(root);info=json.loads((root/'meta/info.json').read_text())
    for f in ('observation.state','action'):
        feature=info['features'][f]
        if feature['dtype']!='float32' or len(feature['shape'])!=1:raise ValueError('Invalid vector feature '+f)
        if not feature.get('names') or len(feature['names'])!=feature['shape'][0]:raise ValueError('Missing feature names')
    if info['total_episodes']<2:raise ValueError('At least two complete episodes are needed for train/validation separation')
    if info['fps']<=0:raise ValueError('Invalid dataset FPS')
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds=LeRobotDataset(repo_id='local/audit',root=root,video_backend='pyav')
    import numpy as np
    for i in range(min(len(ds),10000)):
        frame=ds[i]
        if not np.isfinite(frame['action'].numpy()).all() or not np.isfinite(frame['observation.state'].numpy()).all():raise ValueError('Nonfinite dataset value')
    return {'episodes':info['total_episodes'],'frames':info['total_frames'],'fps':info['fps'],'features':info['features']}
def split(root,path,seed=42,fraction=.2):
    info=audit(root);episodes=list(range(info['episodes']));random.Random(seed).shuffle(episodes)
    n=max(1,round(len(episodes)*fraction));n=min(n,len(episodes)-1)
    result={'schema_version':1,'seed':seed,'dataset_root':str(Path(root).resolve()),
        'dataset_info_sha256':hashlib.sha256((Path(root)/'meta/info.json').read_bytes()).hexdigest(),
        'train':sorted(episodes[n:]),'validation':sorted(episodes[:n])}
    if Path(path).exists():raise FileExistsError('Split exists; preserve reproducible episode assignment')
    Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(result,indent=2));return result
def train(config,split_file,execute=True,resume=False):
    c=yaml.safe_load(Path(config).read_text());s=json.loads(Path(split_file).read_text());root=Path(s['dataset_root'])
    if hashlib.sha256((root/'meta/info.json').read_bytes()).hexdigest()!=s['dataset_info_sha256']:raise ValueError('Dataset changed since split')
    if set(s['train'])&set(s['validation']):raise ValueError('Episode split overlaps')
    c['dataset']['root']=str(root);c['dataset']['episodes']=s['train']
    c['policy']['push_to_hub']=False;c.setdefault('trackio',{})['enable']=False
    c['env_eval_freq']=0
    if c['policy'].get('device','auto')=='auto':
        import torch
        c['policy']['device']='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    args=[sys.executable,'-m','lerobot.scripts.lerobot_train']
    def flatten(v,prefix=''):
        for k,value in v.items():
            key=prefix+'.'+k if prefix else k
            if isinstance(value,dict):yield from flatten(value,key)
            else:yield '--'+key+'='+('null' if value is None else json.dumps(value,separators=(',',':')) if isinstance(value,(list,bool)) else str(value))
    args.extend(flatten(c))
    if resume:
        saved=Path(c['output_dir'])/'checkpoints/last/pretrained_model/train_config.json'
        if not saved.exists():raise FileNotFoundError('Resume checkpoint not found: '+str(saved))
        args=[sys.executable,'-m','lerobot.scripts.lerobot_train','--config_path='+str(saved),'--resume=true','--policy.push_to_hub=false','--trackio.enable=false']
    print('Training command:',subprocess.list2cmdline(args))
    if execute:subprocess.run(args,check=True)
    return args
def export(checkpoint,profile,destination,evaluation=None):
    source=Path(checkpoint).resolve();p=yaml.safe_load(Path(profile).read_text());dest=Path(destination).resolve()
    if dest.exists():raise FileExistsError('Export is immutable; select a new version directory')
    for f in ('config.json','model.safetensors','policy_preprocessor.json','policy_postprocessor.json'):
        if not (source/f).is_file():raise ValueError('Incomplete checkpoint: '+f)
    for field in ('arm_id','state_features','action_features','cameras','action_fps','language_conditioned','calibration_revision'):
        if field not in p:raise ValueError('Missing compatibility field '+field)
    if p['action_fps']<=0:raise ValueError('Invalid policy cadence')
    dest.mkdir(parents=True);shutil.copytree(source,dest/'pretrained_model')
    manifest={**p,'schema_version':1,'lerobot_version':'0.7.0','evaluation':json.loads(Path(evaluation).read_text()) if evaluation else {'task_outcome':'NOT_PHYSICALLY_EVALUATED'}}
    manifest['files']={str(f.relative_to(dest)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (dest/'pretrained_model').rglob('*') if f.is_file()}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2));return manifest
def evaluate(checkpoint,root,repo_id,split_file,output,device='cpu',max_frames=200):
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class,make_pre_post_processors
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    s=json.loads(Path(split_file).read_text());cfg=PreTrainedConfig.from_pretrained(checkpoint);cfg.device=device
    policy=get_policy_class(cfg.type).from_pretrained(checkpoint,config=cfg).to(device).eval()
    pre,post=make_pre_post_processors(cfg,pretrained_path=checkpoint,preprocessor_overrides={'device_processor':{'device':device}})
    ds=LeRobotDataset(repo_id=repo_id,root=root,episodes=s['validation'],video_backend='pyav');squared=0.;count=0;previous_episode=None
    with torch.inference_mode():
        for i in range(min(max_frames,len(ds))):
            f=ds[i];episode=int(f['episode_index'])
            if episode!=previous_episode:policy.reset();pre.reset();post.reset();previous_episode=episode
            batch={k:v for k,v in f.items() if k.startswith('observation.')};batch['task']=f.get('task','')
            action=post(policy.select_action(pre(batch))).cpu().reshape(-1);truth=f['action'].cpu().reshape(-1)
            squared+=float(((action-truth)**2).sum());count+=truth.numel()
    report={'metric':'held_out_action_mse','mse':squared/max(count,1),'scalar_values':count,'validation_episodes':s['validation'],
        'task_outcome':'NOT_PHYSICALLY_EVALUATED','note':'Offline action error does not establish manipulation success'}
    Path(output).write_text(json.dumps(report,indent=2));return report
