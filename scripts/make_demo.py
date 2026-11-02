"""生成教学数据；这些记录不来自 Criteo。"""
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(42)
x = rng.normal(size=(20000, 12))
t = rng.binomial(1, .5, len(x))
p0 = 1 / (1 + np.exp(2 - .7*x[:, 0]))
effect = .18 / (1 + np.exp(-2*x[:, 1]))
df = pd.DataFrame(x, columns=[f'f{i}' for i in range(12)])
df['treatment'] = t
df['conversion'] = rng.binomial(1, p0 + t*effect)
(ROOT/'data').mkdir(exist_ok=True)
df.to_csv(ROOT/'data/demo.csv', index=False)
print('Created 20,000 SYNTHETIC teaching records.')
