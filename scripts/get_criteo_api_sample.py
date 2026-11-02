"""网络受限时从官方 HF 数据集查看 API 随机获取分散的 100 行区块。"""
from pathlib import Path
import argparse, json, time
import numpy as np
import pandas as pd
import requests
ROOT = Path(__file__).resolve().parents[1]
folder = ROOT/'data'
folder.mkdir(exist_ok=True)
cache = folder/'api_blocks'
cache.mkdir(exist_ok=True)
total = 13979592
# Sample 500 nonoverlapping blocks without replacement. Exclude final 92 rows.
blocks = np.random.default_rng(42).choice(total//100, 500, replace=False)
parser = argparse.ArgumentParser()
parser.add_argument('--cached-only', action='store_true', help='Finalize only already downloaded blocks; label as partial pilot')
args = parser.parse_args()
def fetch(block):
    file = cache/f'{block}.json'
    if file.exists(): return json.loads(file.read_text())
    params = dict(dataset='criteo/criteo-uplift',config='default',split='train',offset=int(block)*100,length=100)
    for attempt in range(3):
        try:
            r = requests.get('https://datasets-server.huggingface.co/rows', params=params, timeout=(15,45))
            if r.status_code == 429:
                raise RuntimeError('Dataset service rate limit reached. Stop and retry later; cached blocks are retained.')
            r.raise_for_status()
            rows = [x['row'] for x in r.json()['rows']]
            if len(rows)!=100: raise ValueError('Incomplete block')
            file.write_text(json.dumps(rows))
            return rows
        except RuntimeError:
            raise
        except Exception:
            if attempt==2: raise
            time.sleep(min(30,2**attempt))
rows = []
if not args.cached_only:
    for i, block in enumerate(blocks, 1):
        rows.extend(fetch(int(block)))
        if i%25==0: print(f'Downloaded {i*100:,}/50,000 real records',flush=True)
else:
    blocks = [b for b in blocks if (cache/f'{b}.json').exists()]
    if len(blocks)<10: raise RuntimeError('Too few cached blocks for pilot')
# Sorting removes completion-order nondeterminism.
rows = [dict(row, _sample_block=int(b)) for b in blocks for row in json.loads((cache/f'{b}.json').read_text())]
pd.DataFrame(rows).to_csv(folder/'criteo.csv',index=False)
meta = dict(source='https://huggingface.co/datasets/criteo/criteo-uplift',
            endpoint='https://datasets-server.huggingface.co/rows', sample_rows=len(rows),seed=42,
            sampling=f'{len(blocks)} retrieved randomly selected nonoverlapping 100-row blocks; final 92 rows excluded',
            status='Partial API retrieval pilot due to rate limit' if args.cached_only else 'Complete 500-block sample',
            limitation='Cluster sample, grouped train/test split and cluster-level approximate intervals; exploratory only.',
            block_ids=[int(b) for b in blocks])
(folder/'criteo_source.json').write_text(json.dumps(meta,indent=2))
print('Saved real Criteo sample and provenance.',flush=True)
