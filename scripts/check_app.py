"""运行真实 Streamlit 组件并检查预算交互，无需手工点击。"""
from pathlib import Path
from streamlit.testing.v1 import AppTest
import pandas as pd
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
app = AppTest.from_file(str(ROOT/'app.py')).run(timeout=30)
assert not app.exception, app.exception
assert int(app.metric[0].value.replace(',', '')) > 0
assert len(app.dataframe[1].value) == 3
assert set(app.dataframe[1].value['budget']) == {.2}
app.slider[0].set_value(50).run()
assert not app.exception, app.exception
assert set(app.dataframe[1].value['budget']) == {.5}
for dataset in app.selectbox[0].options:
    app.selectbox[0].select(dataset).run()
    assert not app.exception, app.exception
    assert len(app.dataframe[1].value) == 3
    policies = pd.read_csv(ROOT/'results'/dataset/'policies.csv')
    assert len(policies) == 60 and np.isfinite(policies[['gain','lower','upper']].to_numpy()).all()
    assert policies.groupby('strategy').size().eq(20).all()
    assert policies.loc[policies.budget == 1, 'gain'].max() - policies.loc[policies.budget == 1, 'gain'].min() < 1e-10
print('PASS: both datasets rendered, three policies shown, budget interaction updated results.')
