"""
src/forecast.py
===============
Load the trained model and generate future sales forecasts.

Business Context:
-----------------
After training on historical data (2014–2017), this module builds a synthetic
future date range (next 30, 60, or 90 days beyond the last known date) and
constructs the same time-based features the model learned from.  The result is
a day-by-day revenue forecast that a store manager can use for:

  • Inventory planning  – stock up before high-forecast weeks.
  • Staffing            – schedule extra staff on high-demand forecast days.
  • Cash-flow budgeting – anticipate revenue shortfalls in low-forecast periods.

Author: Paullesley Allwyn
Project: Future Interns – ML Task 1: Sales & Demand Forecasting
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

# Re-use the column lists defined in train_model.py
FEATURE_COLS    = [
    "Year", "Month", "Day", "DayOfWeek", "DayOfYear",
    "WeekOfYear", "Quarter", "Is_Weekend",
    "Month_Sin", "Month_Cos", "Days_Since_Start",
]
DATE_COL        = "Order Date"
TARGET_COL      = "Sales"
MODEL_PATH      = Path(__file__).resolve().parent.parent / "models" / "sales_forecast_model.pkl"


def load_model(path: Path = None) -> RandomForestRegressor:
    """
    Load the persisted RandomForestRegressor from disk.

    Parameters
    ----------
    path : Path, optional
        Override the default model path.

    Returns
    -------
    RandomForestRegressor
    """
    import joblib
    if path is None:
        path = MODEL_PATH
    model = joblib.load(path)
    print(f"  [INFO] Loaded model from {path}")
    return model


def build_future_features(last_date: pd.Timestamp,
                           horizon_days: int,
                           min_date: pd.Timestamp) -> pd.DataFrame:
    """
    Construct a feature DataFrame for future dates the model has never seen.

    The function mirrors exactly the feature engineering done in preprocessing.py
    so the model receives inputs in the same format it was trained on.

    Parameters
    ----------
    last_date     : pd.Timestamp  – last date in the training data.
    horizon_days  : int           – number of future days to forecast (30/60/90).
    min_date      : pd.Timestamp  – earliest date in the entire dataset
                                    (used to compute Days_Since_Start correctly).

    Returns
    -------
    pd.DataFrame
        One row per future day, with all FEATURE_COLS populated.
    """
    # Create the future date range starting the day AFTER the last known date
    future_dates = pd.date_range(
        start = last_date + pd.Timedelta(days=1),
        periods = horizon_days,
        freq = "D",
    )

    df = pd.DataFrame({DATE_COL: future_dates})

    dt = df[DATE_COL]
    df["Year"]             = dt.dt.year
    df["Month"]            = dt.dt.month
    df["Day"]              = dt.dt.day
    df["DayOfWeek"]        = dt.dt.dayofweek
    df["DayOfYear"]        = dt.dt.dayofyear
    df["WeekOfYear"]       = dt.dt.isocalendar().week.astype(int)
    df["Quarter"]          = dt.dt.quarter
    df["Is_Weekend"]       = (dt.dt.dayofweek >= 5).astype(int)
    df["Month_Sin"]        = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_Cos"]        = np.cos(2 * np.pi * df["Month"] / 12)
    df["Days_Since_Start"] = (dt - min_date).dt.days

    return df


def generate_forecast(daily_df: pd.DataFrame,
                       horizon_days: int = 90,
                       model: RandomForestRegressor = None) -> pd.DataFrame:
    """
    Generate a forward-looking sales forecast for the next N days.

    Parameters
    ----------
    daily_df      : pd.DataFrame
        Preprocessed daily DataFrame (used to obtain last_date and min_date).
    horizon_days  : int, optional
        Number of days to forecast ahead (default 90 = ~3 months).
    model         : RandomForestRegressor, optional
        Pre-loaded model. If None, the model is loaded from disk.

    Returns
    -------
    pd.DataFrame
        Forecast DataFrame with columns:
          - ``Order Date``    – future calendar date
          - ``Forecasted_Sales`` – model prediction for that day
    """
    print("\n" + "="*60)
    print("  STAGE 3: FUTURE SALES FORECAST")
    print("="*60)

    if model is None:
        model = load_model()

    last_date = daily_df[DATE_COL].max()
    min_date  = daily_df[DATE_COL].min()

    print(f"  [INFO] Last known date: {last_date.date()}")
    print(f"  [INFO] Forecasting {horizon_days} days ahead …")

    future_feats = build_future_features(last_date, horizon_days, min_date)
    X_future     = future_feats[FEATURE_COLS].values

    y_future     = model.predict(X_future)

    # Clip negative predictions (sales cannot be negative)
    y_future = np.clip(y_future, a_min=0, a_max=None)

    forecast_df = pd.DataFrame({
        DATE_COL:           future_feats[DATE_COL],
        "Forecasted_Sales": y_future,
    })

    total_forecast = forecast_df["Forecasted_Sales"].sum()
    avg_daily      = forecast_df["Forecasted_Sales"].mean()
    peak_day       = forecast_df.loc[forecast_df["Forecasted_Sales"].idxmax(), DATE_COL]
    peak_sales     = forecast_df["Forecasted_Sales"].max()

    print(f"\n  ── {horizon_days}-Day Forecast Summary ─────────────────")
    print(f"  Total forecasted revenue : ${total_forecast:,.2f}")
    print(f"  Average daily sales      : ${avg_daily:,.2f}")
    print(f"  Peak sales day           : {peak_day.date()} (${peak_sales:,.2f})")
    print("  ────────────────────────────────────────────────")
    print(f"\n  ✅ Forecast generated for {horizon_days} days.")

    return forecast_df


if __name__ == "__main__":
    from preprocessing import preprocess_pipeline
    daily = preprocess_pipeline()
    fc    = generate_forecast(daily, horizon_days=90)
    print(fc.head(10))
