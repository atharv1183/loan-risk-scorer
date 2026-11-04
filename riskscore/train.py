"""Train and compare two default-risk models, pick a lending cut-off, and save everything the API needs.

Steps:
1. Split 80/20 (stratified). The 20% test set is touched only once, at the end.
2. 5-fold cross-validation on the training set for a logistic-regression baseline and LightGBM.
3. Choose the approval cut-off on out-of-fold training predictions, using a simple profit rule.
4. Report AUC, Gini, KS, Brier and the business result of the cut-off on the test set.
5. Check fairness by sex (not a model input) and save the model, metrics and charts.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from riskscore.features import FEATURES, build_features

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CLEAN_CSV = ROOT / "data" / "clean" / "credit_default.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
REPORTS = ROOT / "reports"
SEED = 42

# Profit assumption for choosing the cut-off: approving a client who repays earns 1 unit,
# approving a client who defaults loses 5 units. Rejected clients earn nothing.
GAIN_GOOD, LOSS_BAD = 1.0, 5.0

# Risk bands by predicted probability of default.
BANDS = [("A", 0.10), ("B", 0.20), ("C", 0.35), ("D", 0.55), ("E", 1.01)]

# Being later on payments must never lower the predicted risk: this keeps the model sensible
# and easy to explain to a credit team.
MONOTONE_UP = {"months_late_1", "months_late_2", "months_late_3",
               "max_months_late", "times_late_6m", "late_streak", "zero_payment_months"}

LGB_PARAMS = dict(
    n_estimators=450, learning_rate=0.03, num_leaves=15, min_child_samples=60,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    monotone_constraints=[1 if f in MONOTONE_UP else 0 for f in FEATURES],
    random_state=SEED, verbose=-1,
)


def ks_stat(y: np.ndarray, p: np.ndarray) -> float:
    """Kolmogorov-Smirnov: the largest gap between the score distributions of defaulters and payers."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def scores(y: np.ndarray, p: np.ndarray) -> dict:
    auc = roc_auc_score(y, p)
    return {"auc": round(auc, 4), "gini": round(2 * auc - 1, 4), "ks": round(ks_stat(y, p), 4),
            "pr_auc": round(average_precision_score(y, p), 4), "brier": round(brier_score_loss(y, p), 4)}


def profit(y: np.ndarray, approve: np.ndarray) -> float:
    """Average profit per applicant under the GAIN_GOOD / LOSS_BAD rule."""
    return float(np.mean(np.where(approve, np.where(y == 1, -LOSS_BAD, GAIN_GOOD), 0.0)))


def best_cutoff(y: np.ndarray, p: np.ndarray) -> float:
    grid = np.round(np.arange(0.05, 0.80, 0.01), 2)
    return float(max(grid, key=lambda c: profit(y, p < c)))


def band_of(p: float) -> str:
    return next(name for name, upper in BANDS if p < upper)


def logistic_model() -> Pipeline:
    numeric = [f for f in FEATURES if f != "education"]
    prep = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("edu", OneHotEncoder(handle_unknown="ignore"), ["education"]),
    ])
    return Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=2000))])


def main() -> None:
    df = pd.read_csv(CLEAN_CSV)
    X, y = build_features(df), df["default"].to_numpy()
    idx_train, idx_test = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y, random_state=SEED)
    X_tr, X_te, y_tr, y_te = X.iloc[idx_train], X.iloc[idx_test], y[idx_train], y[idx_test]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    models = {"logistic_regression": logistic_model(), "lightgbm": lgb.LGBMClassifier(**LGB_PARAMS)}
    results, oof = {}, {}
    for name, model in models.items():
        oof[name] = cross_val_predict(model, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        fold_auc = [roc_auc_score(y_tr[v], oof[name][v]) for _, v in cv.split(X_tr, y_tr)]
        model.fit(X_tr, y_tr)
        p_te = model.predict_proba(X_te)[:, 1]
        results[name] = {"cv_auc_mean": round(float(np.mean(fold_auc)), 4),
                         "cv_auc_std": round(float(np.std(fold_auc)), 4),
                         "test": scores(y_te, p_te)}
        print(f"{name:20s} CV AUC {np.mean(fold_auc):.4f} ± {np.std(fold_auc):.4f} | test {results[name]['test']}")

    model = models["lightgbm"]
    p_te = model.predict_proba(X_te)[:, 1]

    # Cut-off chosen on out-of-fold training predictions, then judged on the untouched test set.
    cutoff = best_cutoff(y_tr, oof["lightgbm"])
    approve = p_te < cutoff
    late2_rule = df["pay_1"].to_numpy()[idx_test] < 2  # simple rule: reject if 2+ months late now
    business = {
        "assumption": f"approve a payer = +{GAIN_GOOD:g}, approve a defaulter = -{LOSS_BAD:g}, reject = 0",
        "cutoff_probability": cutoff,
        "model": {"approval_rate": round(float(approve.mean()), 4),
                  "default_rate_among_approved": round(float(y_te[approve].mean()), 4),
                  "profit_per_applicant": round(profit(y_te, approve), 4)},
        "approve_everyone": {"approval_rate": 1.0, "default_rate_among_approved": round(float(y_te.mean()), 4),
                             "profit_per_applicant": round(profit(y_te, np.ones_like(y_te, bool)), 4)},
        "rule_reject_2_months_late": {"approval_rate": round(float(late2_rule.mean()), 4),
                                      "default_rate_among_approved": round(float(y_te[late2_rule].mean()), 4),
                                      "profit_per_applicant": round(profit(y_te, late2_rule), 4)},
    }
    business["defaulters_rejected_pct"] = round(float((~approve & (y_te == 1)).sum() / (y_te == 1).sum()), 4)
    business["payers_rejected_pct"] = round(float((~approve & (y_te == 0)).sum() / (y_te == 0).sum()), 4)

    bands = pd.DataFrame({"band": [band_of(p) for p in p_te], "y": y_te})
    band_table = (bands.groupby("band")["y"].agg(clients="size", default_rate="mean").round(4)
                  .reset_index().to_dict(orient="records"))

    sex = df["sex"].to_numpy()[idx_test]  # 1 = male, 2 = female in the source data
    fairness = {lab: {"clients": int((sex == code).sum()),
                      "auc": round(roc_auc_score(y_te[sex == code], p_te[sex == code]), 4),
                      "approval_rate": round(float(approve[sex == code].mean()), 4),
                      "actual_default_rate": round(float(y_te[sex == code].mean()), 4)}
                for lab, code in (("male", 1), ("female", 2))}

    # Global importance = mean absolute contribution (SHAP values computed by LightGBM itself).
    contrib = model.booster_.predict(X_te, pred_contrib=True)[:, :-1]
    importance = (pd.Series(np.abs(contrib).mean(axis=0), index=FEATURES)
                  .sort_values(ascending=False).round(4))

    metrics = {
        "data": {"rows": len(df), "train_rows": len(idx_train), "test_rows": len(idx_test),
                 "default_rate": round(float(y.mean()), 4), "features": len(FEATURES)},
        "models": results, "business": business, "risk_bands": band_table,
        "fairness_by_sex": fairness, "feature_importance": importance.to_dict(),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame({"default": y_te, "probability": p_te.round(5)}).to_csv(REPORTS / "test_scores.csv", index=False)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "cutoff": cutoff, "bands": BANDS,
                 "test_auc": results["lightgbm"]["test"]["auc"]}, MODEL_PATH)
    save_charts(y_te, {n: models[n].predict_proba(X_te)[:, 1] for n in models}, importance)
    print(json.dumps(business, indent=2))
    print("bands", band_table)
    print("fairness", fairness)
    print("top features", importance.head(8).to_dict())


def save_charts(y: np.ndarray, probs: dict, importance: pd.Series) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
        frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        axes[1].plot(mean_pred, frac, marker="o", label=name)
    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[0].set(title="ROC curve (test set)", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].plot([0, 1], [0, 1], "--", color="grey")
    axes[1].set(title="Calibration (test set)", xlabel="Predicted default probability", ylabel="Actual default rate")
    importance.head(10)[::-1].plot.barh(ax=axes[2], color="#1A2D4B")
    axes[2].set(title="Top features (mean |SHAP|)", xlabel="Mean absolute contribution (log-odds)")
    for ax in axes[:2]:
        ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(REPORTS / "model_charts.png", dpi=130)


if __name__ == "__main__":
    main()
