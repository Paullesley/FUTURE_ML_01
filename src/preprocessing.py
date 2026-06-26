"""
src/preprocessing.py
====================
Data loading, cleaning, and feature engineering for the Superstore Sales dataset.

Business Context:
-----------------
The Superstore dataset contains ~9,994 individual order line-items spanning 2014-2017.
For forecasting purposes we aggregate these transactions to a DAILY level so the model
can learn time-based patterns (day-of-week effects, monthly seasonality, annual growth).

Author: Paullesley Allwyn
Project: Future Interns – ML Task 1: Sales & Demand Forecasting
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants – column names discovered during dataset inspection
# ---------------------------------------------------------------------------
DATE_COL    = "Order Date"   # Original date column in the raw CSV
TARGET_COL  = "Sales"        # Revenue column we aim to forecast
CATEGORY_COL = "Category"
REGION_COL   = "Region"
SEGMENT_COL  = "Segment"


def load_raw_data(filepath: str = None) -> pd.DataFrame:
    """
    Load the raw Superstore CSV file with automatic encoding detection.

    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file. Defaults to ``data/sales_data.csv`` relative
        to the project root.

    Returns
    -------
    pd.DataFrame
        Raw, unprocessed DataFrame.
    """
    if filepath is None:
        # Build path relative to the project root (one level up from src/)
        base_dir = Path(__file__).resolve().parent.parent
        filepath = base_dir / "data" / "sales_data.csv"

    print(f"  [INFO] Loading raw data from: {filepath}")
    df = pd.read_csv(filepath, encoding="utf-8")
    print(f"  [INFO] Raw shape: {df.shape}  |  Columns: {list(df.columns)}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise and clean the raw DataFrame.

    Steps
    -----
    1. Strip leading/trailing whitespace from column names.
    2. Parse the date column into proper ``datetime`` objects.
    3. Cast numeric columns (Sales, Quantity, Profit, Discount) to float.
    4. Drop duplicate rows.
    5. Drop rows where the target column (Sales) is null or non-positive.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from :func:`load_raw_data`.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for feature engineering.
    """
    print("  [INFO] Cleaning data …")

    # 1. Normalise column names
    df.columns = df.columns.str.strip()

    # 2. Parse Order Date – format detected as M/D/YYYY (e.g. 11/8/2016)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="%m/%d/%Y", errors="coerce")

    # 3. Coerce numeric columns
    numeric_cols = ["Sales", "Quantity", "Profit", "Discount"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"  [INFO] Dropped {before - len(df)} duplicate rows.")

    # 5. Drop rows with missing/invalid Sales values
    df = df.dropna(subset=[DATE_COL, TARGET_COL])
    df = df[df[TARGET_COL] > 0]  # Sales must be positive

    print(f"  [INFO] Clean shape: {df.shape}")
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create rich time-based and business features from the date column.

    Business Reasoning
    ------------------
    Tree-based models like RandomForest cannot natively understand continuous
    time — they need numeric representations of temporal structure.  The
    features below capture:

    * **Calendar decomposition** – Year, Month, Day, DayOfWeek let the model
      detect annual growth, monthly peaks (e.g. holiday season), and within-
      week patterns.
    * **Is_Weekend** – Helps the model distinguish weekend and weekday sales patterns.
    * **Quarter** – Captures broad seasonal quadrants (Q4 typically strong).
    * **Month_Sin / Month_Cos** – Cyclical encoding ensures December (12) is
      treated as close to January (1), preserving the circular nature of months.
    * **Days_Since_Start** – A monotonically increasing trend feature that
      lets the model capture overall business growth over the 4-year period.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame containing ``Order Date``.

    Returns
    -------
    pd.DataFrame
        DataFrame with all new feature columns appended.
    """
    print("  [INFO] Engineering time-based features …")

    df = df.copy()

    dt = df[DATE_COL]

    df["Year"]          = dt.dt.year
    df["Month"]         = dt.dt.month
    df["Day"]           = dt.dt.day
    df["DayOfWeek"]     = dt.dt.dayofweek          # 0=Monday … 6=Sunday
    df["DayOfYear"]     = dt.dt.dayofyear
    df["WeekOfYear"]    = dt.dt.isocalendar().week.astype(int)
    df["Quarter"]       = dt.dt.quarter
    df["Is_Weekend"]    = (dt.dt.dayofweek >= 5).astype(int)

    # Cyclical month encoding (avoids ordinal jump 12→1)
    df["Month_Sin"]     = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_Cos"]     = np.cos(2 * np.pi * df["Month"] / 12)

    # Days since the very first order – captures long-term growth trend
    min_date            = dt.min()
    df["Days_Since_Start"] = (dt - min_date).dt.days

    print("  [INFO] Features added: Year, Month, Day, DayOfWeek, DayOfYear, "
          "WeekOfYear, Quarter, Is_Weekend, Month_Sin, Month_Cos, Days_Since_Start")
    return df


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate individual order line-items to a **daily** time series.

    Business Reasoning
    ------------------
    Forecasting models benefit from a consistent time grain.  Daily aggregation:

    * Smooths out intra-day noise.
    * Produces one row per calendar day, making the time axis uniform.
    * Allows direct comparison of Sales across days, weeks, and months.

    Missing calendar days (days with no orders) are filled with zero sales,
    because a store open but receiving no orders is a meaningful signal.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame.

    Returns
    -------
    pd.DataFrame
        Daily aggregated DataFrame sorted by date, with one row per day.
    """
    print("  [INFO] Aggregating to daily level …")

    # Sum sales per day; keep the first value of stable time features
    agg_dict = {
        TARGET_COL: "sum",
        "Quantity":  "sum",
        "Profit":    "sum",
        # Time features are the same for all orders on the same day
        "Year":           "first",
        "Month":          "first",
        "Day":            "first",
        "DayOfWeek":      "first",
        "DayOfYear":      "first",
        "WeekOfYear":     "first",
        "Quarter":        "first",
        "Is_Weekend":     "first",
        "Month_Sin":      "first",
        "Month_Cos":      "first",
        "Days_Since_Start": "first",
    }

    daily = df.groupby(DATE_COL).agg(agg_dict).reset_index()
    daily = daily.sort_values(DATE_COL).reset_index(drop=True)

    # Fill any calendar gaps with zero-sales days
    full_range = pd.date_range(daily[DATE_COL].min(), daily[DATE_COL].max(), freq="D")
    daily = (
        daily
        .set_index(DATE_COL)
        .reindex(full_range)
        .rename_axis(DATE_COL)
        .reset_index()
    )
    # Re-compute time features for filled gap days
    daily = daily.fillna(0)
    gap_mask = daily["Year"] == 0
    if gap_mask.any():
        dt = daily.loc[gap_mask, DATE_COL]
        daily.loc[gap_mask, "Year"]           = dt.dt.year
        daily.loc[gap_mask, "Month"]          = dt.dt.month
        daily.loc[gap_mask, "Day"]            = dt.dt.day
        daily.loc[gap_mask, "DayOfWeek"]      = dt.dt.dayofweek
        daily.loc[gap_mask, "DayOfYear"]      = dt.dt.dayofyear
        daily.loc[gap_mask, "WeekOfYear"]     = dt.dt.isocalendar().week.astype(int)
        daily.loc[gap_mask, "Quarter"]        = dt.dt.quarter
        daily.loc[gap_mask, "Is_Weekend"]     = (dt.dt.dayofweek >= 5).astype(int)
        min_date = daily[DATE_COL].min()
        daily.loc[gap_mask, "Days_Since_Start"] = (dt - min_date).dt.days
        daily.loc[gap_mask, "Month_Sin"]      = np.sin(2 * np.pi * dt.dt.month / 12)
        daily.loc[gap_mask, "Month_Cos"]      = np.cos(2 * np.pi * dt.dt.month / 12)

    print(f"  [INFO] Daily shape: {daily.shape}  |  "
          f"Date range: {daily[DATE_COL].min().date()} → {daily[DATE_COL].max().date()}")
    return daily


def preprocess_pipeline(filepath: str = None) -> pd.DataFrame:
    """
    End-to-end preprocessing pipeline.

    Runs the full sequence:
      load_raw_data → clean_data → engineer_features → aggregate_daily

    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file (forwarded to :func:`load_raw_data`).

    Returns
    -------
    pd.DataFrame
        Clean, feature-rich daily sales DataFrame ready for model training.
    """
    print("\n" + "="*60)
    print("  STAGE 1: DATA PREPROCESSING")
    print("="*60)

    raw    = load_raw_data(filepath)
    clean  = clean_data(raw)
    feats  = engineer_features(clean)
    daily  = aggregate_daily(feats)

    print(f"\n  ✅ Preprocessing complete. Final shape: {daily.shape}")
    return daily


# ---------------------------------------------------------------------------
# Quick-run entry point for standalone testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = preprocess_pipeline()
    print(df.head())
    print(df.dtypes)
