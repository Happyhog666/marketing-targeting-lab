from pathlib import Path
import json
import pandas as pd
import streamlit as st
ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Marketing Targeting Lab', layout='wide')
st.title('Marketing Targeting Lab')
st.caption('购买预测 vs. 广告增量 | Response prediction vs. incremental impact')
available = [x.name for x in (ROOT/'results').iterdir() if (x/'metrics.json').exists()]
dataset = st.sidebar.selectbox('数据来源', available, index=available.index('criteo') if 'criteo' in available else 0)
if dataset == 'demo': st.warning('模拟练习数据：不代表真实 Criteo 或商业效果。')
else: st.info('官方 Criteo v2.1 样本。数据经过非均匀抽样和匿名化，结果仅用于探索性基准比较。抽样方式见下方方法。')
meta = json.loads((ROOT/'results'/dataset/'metrics.json').read_text())
df = pd.read_csv(ROOT/'results'/dataset/'policies.csv')
c1,c2,c3 = st.columns(3)
c1.metric('总样本量', f"{meta['rows']:,}")
c2.metric('独立测试集', f"{meta['test_rows']:,}")
c3.metric('Response ROC AUC', f"{meta['response_auc']:.3f}")
if dataset == 'criteo':
    st.warning(f"测试集中仅有 {meta['test_conversions']} 次转化；样本较小，策略差异可能不稳定，不能据此宣称真实业务提升。")
    st.caption('当前为网络限流后取得的 7,000 条真实数据试运行版；尚未完成原计划的 150,000 条全文件随机抽样。AUC 也可能因事件稀少而不稳定。' if meta['rows']==7000 else meta['provenance']['sampling'])
budget = st.sidebar.slider('触达预算（客户比例 %）', 5,100,20,5)/100
st.subheader('预算与估计增量转化')
st.line_chart(df.pivot(index='budget',columns='strategy',values='gain'))
st.caption('纵轴：每名测试用户的估计增量转化数；横轴：触达比例。Random 为固定种子的随机排序。')
selected = df[(df.budget-budget).abs()<.001].copy()
st.dataframe(selected, hide_index=True, width='stretch')
st.caption('lower / upper：近似 95% 点置信区间，不确定性较大时不能认定某种策略更优。')
st.download_button('下载策略评估结果 CSV', df.to_csv(index=False), 'policies.csv')
with st.expander('小白解释：这三个策略在做什么？', expanded=True):
    st.markdown('**Response**：优先投给预测最可能购买的人。\n\n**Uplift**：分别预测投放和不投放时的购买概率，再按两者差值排序。\n\n**Random**：随机选择客户，作为比较基线。')
with st.expander('方法与限制'):
    st.write('仅使用 f0–f11，排除 treatment、visit、exposure 等可能泄漏信息的变量。70% 训练，30% 测试；API 区块样本按区块分组划分，并按区块计算近似标准误。模型为标准化 Logistic Regression T-Learner。使用训练集 treatment 比例计算逆概率加权的测试集增量。此估计依赖处理分配假设；数据发布时的抽样意味着不能还原真实广告收益。预算比较属于探索性分析，后续应增加验证集、重复划分、稳健性与区间估计。')
    st.json(meta)
