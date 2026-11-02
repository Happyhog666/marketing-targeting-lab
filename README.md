# Marketing Targeting Lab

Compare customers likely to purchase with customers whose purchase probability may increase under advertising. A reproducible Python and Streamlit portfolio project.

**Current status:** local setup and dashboard verified. Synthetic teaching run: 20,000 rows. Real Criteo pilot: 7,000 rows from 70 random 100-row blocks retrieved before API rate limiting; only 23 conversions overall and 8 in the test set. These are pipeline checks, not credible evidence of strategy superiority. The planned 150,000-row whole-file sample remains incomplete because of download/network limits. No public deployment has been performed.

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

The demo is synthetic. Real data come from [Criteo's official dataset](https://huggingface.co/datasets/criteo/criteo-uplift), version 2.1. The full downloader samples uniformly over the complete file with seeded random priorities and writes provenance and a SHA-256 hash. It has not completed on this machine. The API fallback `scripts/get_criteo_api_sample.py` targets 500 randomly selected nonoverlapping blocks of 100 rows (excluding the final 92 rows). It caches progress and stops on rate limiting. Run it later to retry; use `--cached-only` to finalize available blocks as a partial pilot. Raw data are excluded from Git. The current pilot was produced with this fallback.

## Method

- Seed 42, 70/30 train/test split. Synthetic and whole-file samples stratify on treatment/outcome. API samples split by block with GroupShuffleSplit so each block belongs to one partition.
- Standardized logistic regression fitted separately in treatment and control arms (T-Learner).
- Response ranks the treatment-arm predicted probability; uplift ranks the difference between arm predictions; random is a seeded baseline.
- Inputs are only f0–f11. Visit and exposure are excluded to avoid post-treatment leakage.
- Held-out incremental gain uses the transformed outcome Y*(T/e-(1-T)/(1-e)), with e estimated on training data. Cumulative gain is divided by the full test-set size. Approximate pointwise intervals use the standard error within the selected group; API samples instead aggregate policy contributions by held-out block for a cluster-level standard error. Intervals condition on fitted models and omit training/propensity estimation uncertainty.
- The dashboard reports response AUC separately; AUC does not establish uplift performance.

## Limits

This is an exploratory benchmark, not production targeting or proof of causal lift in an advertiser's original population. Criteo's nonuniform subsampling prevents recovering original incrementality. Treatment probability assumptions may not hold within all pooled experimental contexts. Features are anonymous. Small conversion counts, model misspecification and sampling noise can reverse strategy rankings. Budget comparisons are exploratory and intervals are not corrected for multiple comparisons. Add validation data, repeated splits, stronger learners and robust policy evaluation before drawing strong conclusions. No real revenue/cost data are available, so no real ROI claims are made.

## Attribution and data terms

Diemert, E., Betlei, A., Renaudin, C., and Amini, M.-R. (2018). *A Large Scale Benchmark for Uplift Modeling*. AdKDD and TargetAd Workshop.

Dataset: CC BY-NC-SA 4.0. See [official terms and description](https://ailab.criteo.com/criteo-uplift-prediction-dataset/). Do not upload raw data or describe synthetic results as Criteo findings. Any redistribution of data-derived material must follow the applicable dataset terms.

## Deployment preparation

The repository includes aggregate results so the app starts without downloading raw data. For Streamlit deployment, select `app.py`, Python 3.11 and `requirements.txt`. Public GitHub publication and public app deployment require the owner's approval and account sign-in. Never commit credentials, `.venv` or raw data.
