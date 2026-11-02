from pathlib import Path
import json
import pandas as pd
import numpy as np
import streamlit as st
ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Marketing Targeting Lab', layout='wide')
st.title('Marketing Targeting Lab')
st.caption('Response prediction vs. incremental impact')
available = [x.name for x in (ROOT/'results').iterdir() if (x/'metrics.json').exists()]
dataset = st.sidebar.selectbox('Data source', available, index=available.index('criteo') if 'criteo' in available else 0)
if dataset == 'demo': st.warning('Synthetic practice data: these results do not represent Criteo data or real business outcomes.')
else: st.info('Official Criteo v2.1 sample. The data are anonymized and were sampled non-uniformly; results are for exploratory benchmarking. See Methods and limitations for details.')
meta = json.loads((ROOT/'results'/dataset/'metrics.json').read_text())
df = pd.read_csv(ROOT/'results'/dataset/'policies.csv')
c1,c2,c3 = st.columns(3)
c1.metric('Total records', f"{meta['rows']:,}")
c2.metric('Held-out test records', f"{meta['test_rows']:,}")
c3.metric('Response ROC AUC (treated test group)', f"{meta['response_auc']:.3f}")
if dataset == 'criteo':
    st.warning(f"The test set contains only {meta['test_conversions']} conversions. Strategy estimates may be unstable and do not demonstrate a real-world business improvement.")
    source = meta['provenance']
    sample_note = 'The planned 150,000-row whole-file random sample has not yet been completed. ' if meta['rows'] < 150000 else ''
    st.caption(f"Sampling: {source['sampling']}. {source.get('status', '')} {sample_note}AUC may be unstable when conversions are rare.")
budget = st.sidebar.slider('Targeting budget (% of customers)', 5,100,20,5)/100
st.subheader('Budget and estimated incremental conversions')
st.line_chart(df.pivot(index='budget',columns='strategy',values='gain'))
st.caption('Vertical axis: estimated incremental conversions per test-set customer. Horizontal axis: proportion targeted. Random uses a fixed-seed ordering.')
areas = []
for strategy, points in df.groupby('strategy'):
    points = points.sort_values('budget')
    area = np.trapezoid(np.r_[0, points['gain'].to_numpy()], np.r_[0, points['budget'].to_numpy()])
    areas.append({'Strategy': strategy, 'Area under gain curve': area})
with st.expander('Compare overall targeting curves'):
    st.dataframe(pd.DataFrame(areas), hide_index=True, width='stretch')
    st.caption('Area under the displayed incremental-gain curve, approximated with trapezoids from 0% to 100% targeting. This exploratory summary has no uncertainty interval and does not establish a significant difference.')
selected = df[(df.budget-budget).abs()<.001].copy()
st.dataframe(selected, hide_index=True, width='stretch')
st.caption('Lower and upper are approximate pointwise 95% confidence bounds. Uncertainty may be too large to establish that one strategy is better.')
comparison_file = ROOT/'results'/dataset/'uplift_vs_response.csv'
if comparison_file.exists():
    difference = pd.read_csv(comparison_file)
    row = difference[(difference.budget-budget).abs()<.001].iloc[0]
    st.write(f"**Uplift minus Response:** {row['difference']:.5f} incremental conversions per test customer (approximate 95% interval: {row['lower']:.5f} to {row['upper']:.5f}).")
    st.caption('This paired comparison uses the same test customers for both policies. The interval does not include model-training uncertainty or adjust for comparing many budgets.')
st.download_button('Download policy results (CSV)', df.to_csv(index=False), 'policies.csv')
with st.expander('What do the three strategies do?', expanded=True):
    st.markdown('**Response:** targets customers predicted to be most likely to convert.\n\n**Uplift:** ranks customers by the predicted difference between conversion probabilities with and without advertising.\n\n**Random:** selects customers at random as a baseline.')
with st.expander('Methods and limitations'):
    st.write('Models use only f0–f11, excluding treatment, visit, and exposure from the predictors to avoid leakage. The data are split 70% for training and 30% for testing. API samples are split by block, with approximate standard errors calculated by block. A standardized logistic regression T-learner predicts each treatment outcome. Test-set incremental gain uses inverse-probability weights based on the training treatment rate. The estimate depends on treatment-assignment assumptions; the published data sampling prevents recovery of the original advertising returns. Budget comparisons are exploratory and require further validation, repeated splits, and robustness checks.')
    st.json(meta)
