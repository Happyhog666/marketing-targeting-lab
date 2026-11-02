"""运行真实 Streamlit 组件并检查预算交互，无需手工点击。"""
from pathlib import Path
from streamlit.testing.v1 import AppTest
ROOT = Path(__file__).resolve().parents[1]
app = AppTest.from_file(str(ROOT/'app.py')).run(timeout=30)
assert not app.exception, app.exception
assert app.metric[0].value in ['20,000','7,000','50,000']
assert len(app.dataframe[0].value) == 3
assert set(app.dataframe[0].value['budget']) == {.2}
app.slider[0].set_value(50).run()
assert not app.exception, app.exception
assert set(app.dataframe[0].value['budget']) == {.5}
for dataset in app.selectbox[0].options:
    app.selectbox[0].select(dataset).run()
    assert not app.exception, app.exception
    assert len(app.dataframe[0].value) == 3
print('PASS: both datasets rendered, three policies shown, budget interaction updated results.')
