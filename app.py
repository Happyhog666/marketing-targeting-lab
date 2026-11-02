from pathlib import Path
import json
import pandas as pd
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
c3.metric('Response ROC AUC', f"{meta['response_auc']:.3f}")
if dataset == 'criteo':
    st.warning(f"The test set contains only {meta['test_conversions']} conversions. Strategy estimates may be unstable and do not demonstrate a real-world business improvement.")
    st.caption('This pilot uses 7,000 real records retrieved before network rate limiting. The planned 150,000-row random sample has not yet been completed. AUC may also be unstable because conversions are rare.' if meta['rows']==7000 else meta['provenance']['sampling'])
budget = st.sidebar.slider('Targeting budget (% of customers)', 5,100,20,5)/100
st.subheader('Budget and estimated incremental conversions')
st.line_chart(df.pivot(index='budget',columns='strategy',values='gain'))
st.caption('Vertical axis: estimated incremental conversions per test-set customer. Horizontal axis: proportion targeted. Random uses a fixed-seed ordering.')
selected = df[(df.budget-budget).abs()<.001].copy()
st.dataframe(selected, hide_index=True, width='stretch')
st.caption('Lower and upper are approximate pointwise 95% confidence bounds. Uncertainty may be too large to establish that one strategy is better.')
st.download_button('Download policy results (CSV)', df.to_csv(index=False), 'policies.csv')
with st.expander('What do the three strategies do?', expanded=True):
    st.markdown('**Response:** targets customers predicted to be most likely to convert.\n\n**Uplift:** ranks customers by the predicted difference between conversion probabilities with and without advertising.\n\n**Random:** selects customers at random as a baseline.')
with st.expander('Methods and limitations'):
    st.write('Models use only f0–f11, excluding treatment, visit, and exposure from the predictors to avoid leakage. The data are split 70% for training and 30% for testing. API samples are split by block, with approximate standard errors calculated by block. A standardized logistic regression T-learner predicts each treatment outcome. Test-set incremental gain uses inverse-probability weights based on the training treatment rate. The estimate depends on treatment-assignment assumptions; the published data sampling prevents recovery of the original advertising returns. Budget comparisons are exploratory and require further validation, repeated splits, and robustness checks.')
    st.json(meta)
