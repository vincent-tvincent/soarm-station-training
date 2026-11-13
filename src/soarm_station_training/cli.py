import argparse
import json
from . import pipeline
def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('record');a.add_argument('--config',required=True);a.add_argument('--episodes',type=int,default=10);a.add_argument('--seconds',type=float,default=30);a.add_argument('--synthetic',action='store_true')
    a=s.add_parser('audit');a.add_argument('root')
    a=s.add_parser('split');a.add_argument('root');a.add_argument('--output',required=True);a.add_argument('--seed',type=int,default=42)
    a=s.add_parser('train');a.add_argument('--config',required=True);a.add_argument('--split',required=True);a.add_argument('--dry-run',action='store_true');a.add_argument('--resume',action='store_true')
    a=s.add_parser('evaluate');a.add_argument('--checkpoint',required=True);a.add_argument('--root',required=True);a.add_argument('--repo-id',required=True);a.add_argument('--split',required=True);a.add_argument('--output',required=True);a.add_argument('--device',default='cpu');a.add_argument('--max-frames',type=int,default=200)
    a=s.add_parser('export');a.add_argument('--checkpoint',required=True);a.add_argument('--profile',required=True);a.add_argument('--destination',required=True);a.add_argument('--evaluation')
    a=p.parse_args()
    if a.command=='record':
        from .collect import collect
        result=collect(a.config,a.episodes,a.seconds,a.synthetic)
    elif a.command=='audit':result=pipeline.audit(a.root)
    elif a.command=='split':result=pipeline.split(a.root,a.output,a.seed)
    elif a.command=='train':result=pipeline.train(a.config,a.split,not a.dry_run,a.resume)
    elif a.command=='evaluate':result=pipeline.evaluate(a.checkpoint,a.root,a.repo_id,a.split,a.output,a.device,a.max_frames)
    elif a.command=='export':result=pipeline.export(a.checkpoint,a.profile,a.destination,a.evaluation)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
