# Loan Risk Scorer

Predicts the chance that a credit-card client defaults next month, picks a lending cut-off that
maximises profit, and serves the model through an API with plain-English reasons for every decision.

Built on the public **UCI "Default of Credit Card Clients"** dataset: 30,000 clients in Taiwan,
6 months of account history each (April–September 2005), 22.1% of whom defaulted.

![Model charts](reports/model_charts.png)

## Results (6,000 clients held out from training)

| Model | Cross-validated AUC (5 folds) | Test AUC | Gini | KS | Brier |
|---|---|---|---|---|---|
| Logistic regression (baseline) | 0.762 ± 0.009 | 0.743 | 0.487 | 0.386 | 0.140 |
| **LightGBM (monotonic)** | **0.785 ± 0.005** | **0.781** | **0.563** | **0.430** | **0.134** |

**Lending decision.** Assuming an approved payer earns 1 unit and an approved defaulter loses 5:

| Policy | Approved | Default rate among approved | Profit per applicant |
|---|---|---|---|
| Approve everyone | 100% | 22.1% | −0.33 |
| Simple rule: reject if 2+ months late now | 89.4% | 16.6% | +0.01 |
| **Model, cut-off 17%** | **57.6%** | **10.0%** | **+0.23** |

At the cut-off the model turns away 74% of eventual defaulters while turning away 33% of payers.
The riskiest 20% of clients contain 51% of all defaulters (see the gains table in `reports/sql_findings.md`).

**Risk bands** (actual default rate on the test set): A 7.4% · B 14.2% · C 25.4% · D 44.3% · E 68.1%.

**Fairness.** Sex and marital status are not model inputs. Checked afterwards on the test set:
AUC is 0.778 for men and 0.783 for women; approval rates are 56.1% and 58.6%, in line with their actual
default rates (23.4% and 21.3%).

## What the SQL analysis found

Run with DuckDB over the cleaned data (`sql/`, results in `reports/sql_findings.md`):

- Clients **2 months late** last month default at **69%**, against **13%** for clients paying on time.
- Default rate falls steadily with credit limit: **36%** under 50k, **21%** from 50k to 300k, **14%** above 300k.
- Clients over their limit default at 30%, but the 70–100% utilisation band holds more of all defaults (29%).
- The youngest (21–24) and oldest (60+) age bands are the riskiest, 5–6 points above average.

## How it works

1. **`riskscore/data.py`** downloads the data, renames columns and folds undocumented codes.
2. **`riskscore/features.py`** builds 17 features from the raw history: months late (latest, worst,
   count, current streak), utilisation (latest, average, trend), payment-to-bill ratios, and months with
   a bill but no payment. Training and the API call the same function, so features can't drift apart.
3. **`riskscore/train.py`**
   - 80/20 stratified split; the test set is used once at the end.
   - 5-fold CV for logistic regression and LightGBM.
   - **Monotonic constraints:** being later on payments can never lower the predicted risk.
   - The cut-off is chosen on out-of-fold training predictions, not on the test set.
   - Reason codes come from LightGBM's built-in SHAP values.
4. **`riskscore/sql_analysis.py`** runs the SQL files (GROUP BY, CASE bands, HAVING, CTEs, window
   functions such as `RANK`, `NTILE` and running `SUM OVER`).
5. **`api/app.py`** is a FastAPI service:
   - `POST /score` returns probability, band, decision and the top 3 reasons.
   - `POST /score/batch` scores up to 1,000 applicants per call.
   - `GET /metrics` and `GET /health`.
   - A one-page dashboard at `/`.
   - Inputs are checked with Pydantic, and bad requests get a 422.
6. **`tests/`** has 11 pytest tests: feature edge cases (zero and negative bills, streaks), API decisions,
   input validation, and a check that a later payment never lowers risk.

## Run it

```bash
python -m venv .venv && .venv/Scripts/activate      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m riskscore.data          # download + clean
python -m riskscore.train         # train, evaluate, save model and charts
python -m riskscore.sql_analysis  # SQL findings
pytest                            # 11 tests
uvicorn api.app:app --reload      # open http://127.0.0.1:8000
```

Or with Docker (after training): `docker build -t loan-risk . && docker run -p 8000:8000 loan-risk`

Example request:

```bash
curl -X POST http://127.0.0.1:8000/score -H "Content-Type: application/json" -d '{
  "limit_bal": 50000, "age": 26, "education": 2,
  "pay_status": [2, 2, 2, 0, 0, 0],
  "bill_amounts": [49500, 48800, 47100, 45000, 43900, 42000],
  "paid_amounts": [0, 0, 1500, 2000, 1800, 1700]}'
```

## Limits

- The data is from one Taiwanese card issuer in 2005. The model shows the method; it is not a model for
  lending today.
- The 1 : 5 profit rule is an assumption. A real lender would use its own margins and loss-given-default.
- Age is a model input. Some lenders and regulators restrict this; dropping it is a one-line change in
  `features.py`.

Data: Yeh, I-C. (2009). Default of Credit Card Clients. UCI Machine Learning Repository,
https://doi.org/10.24432/C55S3H (CC BY 4.0).
