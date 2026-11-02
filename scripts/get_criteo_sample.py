"""下载官方修订版，分块随机抽样；不只取文件开头。"""
from pathlib import Path
import argparse, hashlib, json
import requests
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--rows', type=int, default=150000)
a = p.parse_args()
if a.rows < 1000: p.error('--rows must be at least 1000')
folder = ROOT/'data'
folder.mkdir(exist_ok=True)
archive = folder/'criteo-v2.1.csv.gz'
url = 'https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz?download=true'
if not archive.exists():
    tmp = archive.with_suffix('.part')
    with requests.get(url, stream=True, timeout=(30,120)) as r:
        r.raise_for_status()
        with tmp.open('wb') as f:
            for chunk in r.iter_content(1024*1024): f.write(chunk)
    tmp.replace(archive)
rng = np.random.default_rng(42)
sample = None
total = 0
for chunk in pd.read_csv(archive, compression='gzip', chunksize=200000):
    total += len(chunk)
    chunk['_key'] = rng.random(len(chunk))
    sample = pd.concat([sample, chunk], ignore_index=True).nsmallest(a.rows, '_key')
    print(f'Scanned {total:,} rows', flush=True)
sample.drop(columns='_key').to_csv(folder/'criteo.csv', index=False)
meta = dict(source=url, rows_scanned=total, sample_rows=len(sample), seed=42,
            sampling='uniform random priorities over entire file',
            sha256=hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest())
(folder/'criteo_source.json').write_text(json.dumps(meta, indent=2))
print(meta)
