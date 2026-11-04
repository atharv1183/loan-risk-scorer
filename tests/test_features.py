import pandas as pd

from riskscore.features import EXCLUDED, FEATURES, build_features


def account(**overrides):
    row = {"limit_bal": 100000, "age": 30, "education": 2, "sex": 1, "marriage": 1}
    for i in range(1, 7):
        row[f"pay_{i}"] = 0
        row[f"bill_{i}"] = 20000
        row[f"paid_{i}"] = 2000
    row.update(overrides)
    return pd.DataFrame([row])


def test_columns_and_no_protected_attributes():
    x = build_features(account())
    assert list(x.columns) == FEATURES
    assert not set(EXCLUDED) & set(x.columns)
    assert not x.isna().any().any()


def test_late_streak_counts_back_from_latest_month():
    x = build_features(account(pay_1=2, pay_2=1, pay_3=0, pay_4=3))
    assert x.loc[0, "late_streak"] == 2
    assert x.loc[0, "times_late_6m"] == 3
    assert x.loc[0, "max_months_late"] == 3


def test_paid_in_full_is_not_late():
    x = build_features(account(**{f"pay_{i}": -1 for i in range(1, 7)}))
    assert x.loc[0, "max_months_late"] == 0
    assert x.loc[0, "late_streak"] == 0


def test_zero_and_negative_bills_do_not_break_ratios():
    bills = {f"bill_{i}": 0 for i in range(1, 7)}
    bills["bill_2"] = -500  # a credit balance (customer overpaid)
    x = build_features(account(**bills))
    assert x.loc[0, "util_1"] == 0
    assert x.loc[0, "pay_ratio_1"] == 0
    assert x.loc[0, "zero_payment_months"] == 0


def test_payment_ratio_matches_previous_bill():
    x = build_features(account(paid_1=10000, bill_2=20000))
    assert x.loc[0, "pay_ratio_1"] == 0.5
