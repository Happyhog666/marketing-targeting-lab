# Marketing Targeting Lab

Compare customers likely to purchase with customers whose purchase probability may increase under advertising. A reproducible Python and Streamlit portfolio project.

**[Explore the live dashboard](https://marketing-targeting-lab-9gpecytcvp8vjs8smg6d4u.streamlit.app/)** | Python · scikit-learn · Streamlit

**Business question:** Under a fixed targeting budget, should marketers prioritize people most likely to convert, or people whose conversion probability may change because of advertising?

The analysis uses a reproducible random sample of **150,000 records** drawn from all 13,979,592 records in Criteo v2.1. The 45,000-record test set contains 127 conversions, including only 12 in the control group. The separate 20,000-record demo dataset is synthetic.

At a 20% targeting budget, the estimated difference between uplift and response targeting is −0.000147 incremental conversions per test customer (approximate pointwise 95% interval −0.000621 to 0.000327). The interval includes zero, so this exploratory run does not establish that either policy is better. These numbers are benchmark estimates, not an advertiser's real-world returns.

## Run locally (Windows, Python 3.11)

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts/make_demo.py
.venv\Scripts\python.exe scripts/train.py
.venv\Scripts\python.exe scripts/get_criteo_sample.py --rows 150000
.venv\Scripts\python.exe scripts/train.py --data criteo
.venv\Scripts\python.exe -m streamlit run app.py
```

The demo is synthetic. Real data come from [Criteo's official dataset](https://huggingface.co/datasets/criteo/criteo-uplift), version 2.1. The downloader verifies the source archive's SHA-256, samples uniformly over the complete file with seeded random priorities, and writes provenance. The API fallback `scripts/get_criteo_api_sample.py` can retrieve a smaller block sample when the full download is unavailable; it was not used for the current published result. Raw data are excluded from Git.

## Method

- Seed 42, 70/30 train/test split. Synthetic and whole-file samples stratify on treatment/outcome. API samples split by block with GroupShuffleSplit so each block belongs to one partition.
- Standardized logistic regression fitted separately in treatment and control arms (T-Learner).
- Response ranks the treatment-arm predicted probability; uplift ranks the difference between arm predictions; random is a seeded baseline.
- Inputs are only f0–f11. Visit and exposure are excluded to avoid post-treatment leakage.
- Held-out incremental gain uses the transformed outcome Y*(T/e-(1-T)/(1-e)), with e estimated on training data. Cumulative gain is divided by the full test-set size. Approximate pointwise intervals use the standard error within the selected group; API samples instead aggregate policy contributions by held-out block for a cluster-level standard error. Intervals condition on fitted models and omit training/propensity estimation uncertainty.
- At each budget, the dashboard also compares uplift against response using a paired difference on the same test set. Its pointwise normal interval likewise omits training uncertainty and any correction for examining multiple budgets. The area under each displayed gain curve is computed by trapezoidal integration, without an uncertainty interval.
- The dashboard reports response AUC on the treated test group separately; AUC does not establish uplift performance.

## Limits

This is an exploratory benchmark, not production targeting or proof of causal lift in an advertiser's original population. Criteo's nonuniform subsampling prevents recovering original incrementality. Treatment probability assumptions may not hold within all pooled experimental contexts. Features are anonymous. Small conversion counts, model misspecification and sampling noise can reverse strategy rankings. Budget comparisons are exploratory and intervals are not corrected for multiple comparisons. Add validation data, repeated splits, stronger learners and robust policy evaluation before drawing strong conclusions. No real revenue/cost data are available, so no real ROI claims are made.

## Attribution and data terms

Diemert, E., Betlei, A., Renaudin, C., and Amini, M.-R. (2018). *A Large Scale Benchmark for Uplift Modeling*. AdKDD and TargetAd Workshop.

Dataset: CC BY-NC-SA 4.0. See [official terms and description](https://ailab.criteo.com/criteo-uplift-prediction-dataset/). Do not upload raw data or describe synthetic results as Criteo findings. Any redistribution of data-derived material must follow the applicable dataset terms.

## Deployment

The repository includes aggregate results so the app starts without downloading raw data. Streamlit Community Cloud runs `app.py` with `requirements.txt`. Never commit credentials, `.venv` or raw data.
