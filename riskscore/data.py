"""Load the UCI credit card default data, clean it, and save a tidy copy.

Source: Yeh, I-C. (2009). "Default of Credit Card Clients", UCI Machine Learning Repository.
30,000 credit card holders in Taiwan, April to September 2005. The target is whether
the client defaulted on the next month's payment (October 2005).
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_ZIP = ROOT / "data" / "raw" / "uci_default.zip"
CLEAN_CSV = ROOT / "data" / "clean" / "credit_default.csv"

# Month 1 is the most recent month (September 2005), month 6 the oldest (April 2005).
# The original file calls the most recent repayment status PAY_0, then PAY_2..PAY_6.
RENAME = {
    "ID": "client_id",
    "LIMIT_BAL": "limit_bal",
    "SEX": "sex",
    "EDUCATION": "education",
    "MARRIAGE": "marriage",
    "AGE": "age",
    "PAY_0": "pay_1",
    **{f"PAY_{i}": f"pay_{i}" for i in range(2, 7)},
    **{f"BILL_AMT{i}": f"bill_{i}" for i in range(1, 7)},
    **{f"PAY_AMT{i}": f"paid_{i}" for i in range(1, 7)},
    "default payment next month": "default",
}


SOURCE_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"


def download() -> None:
    """Fetch the dataset from the UCI repository (CC BY 4.0) if it is not already on disk."""
    if RAW_ZIP.exists():
        return
    import urllib.request

    RAW_ZIP.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {SOURCE_URL}")
    urllib.request.urlretrieve(SOURCE_URL, RAW_ZIP)


def load_raw() -> pd.DataFrame:
    download()
    with zipfile.ZipFile(RAW_ZIP) as z:
        with z.open("default of credit card clients.xls") as f:
            # Row 0 holds X1..X23 labels; row 1 holds the real column names.
            return pd.read_excel(f, header=1)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=RENAME)
    # Education codes 0, 5 and 6 are undocumented; fold them into 4 ("other").
    df["education"] = df["education"].where(df["education"].isin([1, 2, 3, 4]), 4)
    # Marriage code 0 is undocumented; fold it into 3 ("other").
    df["marriage"] = df["marriage"].where(df["marriage"].isin([1, 2, 3]), 3)
    # Repayment status: -2 no use, -1 paid in full, 0 revolving, 1..9 months late.
    # Treat everything that is not late as 0 so the column means "months late".
    for i in range(1, 7):
        df[f"months_late_{i}"] = df[f"pay_{i}"].clip(lower=0)
    return df


def main() -> None:
    df = clean(load_raw())
    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_CSV, index=False)
    print(f"Saved {len(df):,} rows to {CLEAN_CSV.relative_to(ROOT)}; default rate {df['default'].mean():.1%}")


if __name__ == "__main__":
    main()
