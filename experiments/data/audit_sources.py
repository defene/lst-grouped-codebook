"""Recompute source counts and cross-source tile/split intersections, read-only."""
from pathlib import Path
from collections import defaultdict, Counter
import csv, hashlib, json

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
PRITHVI=ROOT/'data_inspection/prithvi/inventory.jsonl'
LSTSR=ROOT/'data_inspection/extracted_metadata/lstsr_tb_final_dataset_20260719/phase4_dataset_manifest.csv'

def main():
    tiles={name:defaultdict(set) for name in ['prithvi','lstsr_tb']}
    counts={name:Counter() for name in tiles}
    months=Counter()
    with PRITHVI.open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            tiles['prithvi'][r['split']].add(r['tile'].removeprefix('T'))
            counts['prithvi'][r['split']]+=1
    with LSTSR.open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            tiles['lstsr_tb'][r['split']].add(r['mgrs_tile'].removeprefix('T'))
            counts['lstsr_tb'][r['split']]+=1
            months[r['split']]+=int(r['available_month_count'])
    intersections={a:{b:sorted(x&y) for b,y in tiles['lstsr_tb'].items()} for a,x in tiles['prithvi'].items()}
    def digest(p):
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        return h.hexdigest()
    report={
        'source_manifests':{str(p.relative_to(ROOT)).replace('\\','/') : digest(p) for p in [PRITHVI,LSTSR]},
        'record_counts':{k:dict(v) for k,v in counts.items()},
        'lstsr_usable_month_counts':dict(months),
        'tile_counts':{k:{s:len(v) for s,v in d.items()} for k,d in tiles.items()},
        'shared_tile_ids':len(set.union(*tiles['prithvi'].values())&set.union(*tiles['lstsr_tb'].values())),
        'cross_split_tile_counts':{a:{b:len(v) for b,v in d.items()} for a,d in intersections.items()},
        'cross_split_tile_ids':intersections,
        'meaning':'Intersect normalized target tile IDs, not verified pixel footprints or duplicate payloads. Prithvi uses the saved file census; LSTSR reads the current extracted canonical manifest.',
    }
    (OUT/'source_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='cross_split_tile_ids'},indent=2))

if __name__=='__main__':main()
