"""Feature engineering shared by training and the API, so both compute features the same way."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Sex and marital status are kept out of the model on purpose: lending decisions should not
# depend on them. They stay in the data only to check the model treats groups fairly.
EXCLUDED = ["sex", "marriage"]

FEATURES = [
    "limit_bal", "age", "education",
    "months_late_1", "months_late_2", "months_late_3",
    "max_months_late", "times_late_6m", "late_streak",
    "util_1", "util_avg_6m", "util_trend",
    "pay_ratio_1", "pay_ratio_avg", "zero_payment_months",
    "bill_avg_6m", "paid_avg_6m",
]


def _safe_ratio(num: pd.Series, den: pd.Series, cap: float) -> pd.Series:
    """num / den, 0 where den <= 0, clipped to [0, cap] so outliers don't dominate."""
    out = np.where(den > 0, num / den.where(den > 0, 1), 0.0)
    return pd.Series(np.clip(out, 0, cap), index=num.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw account history (limit, 6 months of status, bills, payments) into model features."""
    x = pd.DataFrame(index=df.index)
    x["limit_bal"] = df["limit_bal"].astype(float)
    x["age"] = df["age"].astype(float)
    x["education"] = df["education"].astype(int)

    late = pd.DataFrame({i: df[f"pay_{i}"].clip(lower=0) for i in range(1, 7)})
    for i in (1, 2, 3):
        x[f"months_late_{i}"] = late[i]
    x["max_months_late"] = late.max(axis=1)
    x["times_late_6m"] = (late > 0).sum(axis=1)
    # Consecutive late months counting back from the most recent one.
    streak = np.zeros(len(df), dtype=int)
    still = np.ones(len(df), dtype=bool)
    for i in range(1, 7):
        still &= (late[i] > 0).to_numpy()
        streak += still
    x["late_streak"] = streak

    limit = df["limit_bal"].astype(float)
    bills = pd.DataFrame({i: df[f"bill_{i}"].astype(float) for i in range(1, 7)})
    paid = pd.DataFrame({i: df[f"paid_{i}"].astype(float) for i in range(1, 7)})
    utils = pd.DataFrame({i: _safe_ratio(bills[i], limit, cap=3.0) for i in range(1, 7)})
    x["util_1"] = utils[1]
    x["util_avg_6m"] = utils.mean(axis=1)
    x["util_trend"] = utils[1] - utils[6]  # positive = balance growing against the limit

    # Payment made in month i settles the bill of month i+1 (the previous statement).
    ratios = pd.DataFrame({i: _safe_ratio(paid[i], bills[i + 1], cap=2.0) for i in range(1, 6)})
    x["pay_ratio_1"] = ratios[1]
    x["pay_ratio_avg"] = ratios.mean(axis=1)
    x["zero_payment_months"] = ((paid == 0) & (bills.shift(-1, axis=1).fillna(0) > 0)).sum(axis=1)

    x["bill_avg_6m"] = bills.mean(axis=1)
    x["paid_avg_6m"] = paid.mean(axis=1)
    return x[FEATURES]
