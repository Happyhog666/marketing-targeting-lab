"""分别学习投放/未投放的购买概率，在独立测试集比较策略。"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import joblib
ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--data', choices=['demo','criteo'], default='demo')
a = p.parse_args()
df = pd.read_csv(ROOT/f'data/{a.data}.csv')
features = [f'f{i}' for i in range(12)]
assert not df[features+['conversion','treatment']].isna().any().any()
if '_sample_block' in df:
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1,test_size=.3,random_state=42).split(df,groups=df['_sample_block']))
    train, test = df.iloc[train_idx], df.iloc[test_idx]
else:
    train, test = train_test_split(df, test_size=.3, random_state=42,
                                  stratify=df.treatment.astype(str)+df.conversion.astype(str))
models = []
for arm in [0, 1]:
    d = train[train.treatment == arm]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(d[features], d.conversion)
    models.append(model)
p0, p1 = [m.predict_proba(test[features])[:,1] for m in models]
e = float(train.treatment.mean())
# Horvitz-Thompson transformed outcome: estimates benchmark incremental gain.
z = test.conversion.to_numpy()*(test.treatment.to_numpy()/e-(1-test.treatment.to_numpy())/(1-e))
scores = {'Response':p1, 'Uplift':p1-p0, 'Random':np.random.default_rng(42).random(len(test))}
records = []
for name, score in scores.items():
    order = np.argsort(-score, kind='stable')
    for fraction in np.arange(.05, 1.001, .05):
        n = max(2, int(len(test)*fraction))
        values = z[order[:n]]
        gain = float(values.sum()/len(test))
        if '_sample_block' in test:
            contributions = np.zeros(len(test))
            contributions[order[:n]] = z[order[:n]]
            groups = pd.DataFrame({'g':test['_sample_block'].to_numpy(),'v':contributions}).groupby('g')['v'].sum()
            se = float(groups.std(ddof=1)/np.sqrt(len(groups))/100)
        else:
            se = float(values.std(ddof=1)/np.sqrt(n)*n/len(test))
        records.append(dict(strategy=name, budget=round(float(fraction),2),
                            selected=n, gain=gain, lower=gain-1.96*se, upper=gain+1.96*se))
out = ROOT/'results'/a.data
out.mkdir(parents=True, exist_ok=True)
joblib.dump({'control':models[0],'treatment':models[1],'features':features},out/'models.joblib')
pd.DataFrame(records).to_csv(out/'policies.csv', index=False)
comparisons = []
response_order = np.argsort(-scores['Response'], kind='stable')
uplift_order = np.argsort(-scores['Uplift'], kind='stable')
for fraction in np.arange(.05, 1.001, .05):
    n = max(2, int(len(test)*fraction))
    difference = np.zeros(len(test))
    difference[uplift_order[:n]] += z[uplift_order[:n]]
    difference[response_order[:n]] -= z[response_order[:n]]
    estimate = float(difference.mean())
    if '_sample_block' in test:
        sums = pd.DataFrame({'g':test['_sample_block'].to_numpy(),'v':difference}).groupby('g')['v'].sum()
        se = float(sums.std(ddof=1)/np.sqrt(len(sums))/100)
    else:
        se = float(difference.std(ddof=1)/np.sqrt(len(difference)))
    comparisons.append(dict(budget=round(float(fraction),2), difference=estimate,
                            lower=estimate-1.96*se, upper=estimate+1.96*se))
pd.DataFrame(comparisons).to_csv(out/'uplift_vs_response.csv', index=False)
meta = dict(dataset=a.data, rows=len(df), train_rows=len(train), test_rows=len(test),
            conversions=int(df.conversion.sum()), test_conversions=int(test.conversion.sum()),
            test_arm_counts=test.groupby('treatment').conversion.agg(['size','sum']).to_dict(),
            arm_counts=df.groupby('treatment').conversion.agg(['size','sum']).to_dict(),
            treatment_probability=e, response_auc=float(roc_auc_score(test.loc[test.treatment == 1, 'conversion'],p1[test.treatment.to_numpy() == 1])),
            features=features, seed=42,
            caveat='Approximate pointwise intervals; benchmark estimates, not original advertiser ROI. No multiple-testing correction.')
if a.data == 'criteo':
    meta['provenance'] = json.loads((ROOT/'data/criteo_source.json').read_text())
if '_sample_block' in df:
    assert set(train['_sample_block']).isdisjoint(test['_sample_block'])
    meta['split'] = 'GroupShuffleSplit on sampled blocks; 70/30'
    meta['interval'] = 'Approximate cluster-robust pointwise 95% normal interval across held-out blocks'
(out/'metrics.json').write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
