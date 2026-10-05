import argparse
import json
from pathlib import Path
from .core import dataset_name
from .prepare import PROJECT, prepare_dataset

def main():
    parser=argparse.ArgumentParser(description='Unified EO data preparation and reading')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare')
    p.add_argument('--dataset',default='all')
    p.add_argument('--root',type=Path,default=PROJECT)
    p.add_argument('--output',type=Path)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--level',type=int,default=3)
    p.add_argument('--limit',type=int,help='Development subset only; rejected by the default training loader')
    v=sub.add_parser('verify')
    v.add_argument('--root',type=Path,default=PROJECT)
    v.add_argument('--output',type=Path)
    v.add_argument('--dataset',default='all')
    v.add_argument('--samples',type=int,default=16)
    v.add_argument('--allow-development',action='store_true')
    r=sub.add_parser('restore-prithvi')
    r.add_argument('--root',type=Path,default=PROJECT)
    r.add_argument('--prepared',type=Path)
    r.add_argument('--destination',type=Path,required=True)
    r.add_argument('--limit',type=int)
    a=sub.add_parser('archive-prithvi')
    a.add_argument('--root',type=Path,default=PROJECT)
    a.add_argument('--prepared',type=Path)
    a.add_argument('--destination',type=Path,required=True)
    a=sub.add_parser('attach-prithvi')
    a.add_argument('archive',type=Path)
    a.add_argument('--root',type=Path,default=PROJECT)
    a.add_argument('--prepared',type=Path)
    args=parser.parse_args()
    if args.command=='prepare':
        if args.workers<1 or args.workers>32:parser.error('workers must be 1..32')
        if args.limit is not None and args.limit<1:parser.error('limit must be positive')
        output=args.output or args.root/('processed_data/development' if args.limit else 'processed_data/v1')
        names=['prithvi','lstsr_tb'] if args.dataset=='all' else [dataset_name(args.dataset)]
        for name in names:prepare_dataset(args.root.resolve(),name,output.resolve(),args.workers,args.level,args.limit)
    elif args.command=='verify':
        from .verify import verify_all
        print(json.dumps(verify_all(args.root,args.output,args.dataset,args.samples,args.allow_development),indent=2))
    elif args.command=='restore-prithvi':
        from .verify import restore_prithvi
        print(json.dumps(restore_prithvi(args.root,args.prepared,args.destination,args.limit),indent=2))
    elif args.command=='archive-prithvi':
        from .archive import create_archive,attach_archive
        report=create_archive(args.root,args.destination,args.prepared)
        attach_archive(args.root,args.destination,args.prepared,report)
        print(json.dumps(report,indent=2))
    else:
        from .archive import verify_archive,attach_archive
        report=verify_archive(args.archive)
        print(json.dumps(attach_archive(args.root,args.archive,args.prepared,report),indent=2))

if __name__=='__main__':main()
