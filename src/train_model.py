"""
src/train_model.py
==================
Sequential train/test split, RandomForestRegressor training, evaluation, and
model persistence for the Superstore daily sales forecast.

Business Context:
-----------------
We use a SEQUENTIAL (time-ordered) split — the last 20 % of calendar days form
the validation set.  This mirrors real production usage: the model is trained on
the past and evaluated on a future holdout it has never seen.  Using random
splitting would leak future information into training and produce optimistically
biased error metrics.

Author: Paullesley Allwyn
Project: Future Interns – ML Task 1: Sales & Demand Forecasting
"""

import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Feature columns used by the model (must match what preprocessing produces)
FEATURE_COLS = [
    "Year", "Month", "Day", "DayOfWeek", "DayOfYear",
    "WeekOfYear", "Quarter", "Is_Weekend",
    "Month_Sin", "Month_Cos", "Days_Since_Start",
]

TARGET_COL   = "Sales"
DATE_COL     = "Order Date"
MODEL_PATH   = Path(__file__).resolve().parent.parent / "models" / "sales_forecast_model.pkl"


def sequential_split(df: pd.DataFrame, test_ratio: float = 0.20):
    """
    Split the daily time-series DataFrame sequentially (chronological order).

    Parameters
    ----------
    df : pd.DataFrame
        Daily aggregated DataFrame sorted by date.
    test_ratio : float
        Fraction of the total data to reserve for the test/validation set.
        Default 0.20 = last 20 % of calendar days.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    split_idx = int(len(df) * (1 - test_ratio))
    train_df  = df.iloc[:split_idx].copy()
    test_df   = df.iloc[split_idx:].copy()

    print(f"  [INFO] Train: {len(train_df)} days  "
          f"({train_df[DATE_COL].min().date()} → {train_df[DATE_COL].max().date()})")
    print(f"  [INFO] Test : {len(test_df)} days  "
          f"({test_df[DATE_COL].min().date()} → {test_df[DATE_COL].max().date()})")
    return train_df, test_df


def train_model(train_df: pd.DataFrame) -> RandomForestRegressor:
    """
    Train a RandomForestRegressor on the daily sales data.

    Why RandomForest?
    -----------------
    * **Handles non-linearity** – sales patterns are rarely linear.
    * **Robust to outliers** – holiday spikes won't collapse the model.
    * **Built-in feature importance** – gives interpretable business insights.
    * **No feature scaling required** – tree splits are threshold-based.

    Hyperparameters chosen:
    -----------------------
    * n_estimators=300  – enough trees for stable variance reduction.
    * max_depth=15       – prevents over-fitting on 1,400 training days.
    * min_samples_leaf=3 – each leaf covers at least 3 days (avoids noise memorisation).
    * random_state=42    – reproducibility.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training portion of the daily DataFrame.

    Returns
    -------
    RandomForestRegressor
        Fitted model.
    """
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET_COL].values

    print(f"  [INFO] Training RandomForestRegressor on {len(X_train)} samples …")

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=3,
        max_features="sqrt",    # Standard practice for regression RF
        n_jobs=-1,              # Use all available CPU cores
        random_state=42,
    )
    model.fit(X_train, y_train)
    print("  [INFO] Training complete.")
    return model


def evaluate_model(model: RandomForestRegressor,
                   test_df: pd.DataFrame) -> dict:
    """
    Evaluate the trained model on the held-out test set.

    Metrics Explained (for a business audience)
    --------------------------------------------
    * **MAE (Mean Absolute Error)** – on average, predictions are off by
      $X in daily sales.  Easy to communicate to a store manager.
    * **RMSE (Root Mean Squared Error)** – penalises large errors more
      heavily; useful for detecting catastrophic mis-forecasts.
    * **MAPE (Mean Absolute Percentage Error)** – percentage deviation;
      useful when absolute dollar values vary widely.

    Parameters
    ----------
    model : RandomForestRegressor
    test_df : pd.DataFrame

    Returns
    -------
    dict
        Dictionary with keys: predictions, mae, rmse, mape.
    """
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df[TARGET_COL].values

    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # MAPE – avoid division by zero on zero-sales days
    non_zero = y_test != 0
    mape = np.mean(np.abs((y_test[non_zero] - y_pred[non_zero]) / y_test[non_zero])) * 100

    print("\n  ── Evaluation Metrics ──────────────────────────")
    print(f"  MAE  : ${mae:,.2f}  (avg daily prediction error)")
    print(f"  RMSE : ${rmse:,.2f}  (penalises large misses)")
    print(f"  MAPE : {mape:.2f}%  (average % deviation)")
    print("  ────────────────────────────────────────────────")

    return {
        "predictions": y_pred,
        "mae":  mae,
        "rmse": rmse,
        "mape": mape,
    }


def save_model(model: RandomForestRegressor,
               path: Path = None) -> None:
    """
    Persist the trained model to disk using joblib.

    Parameters
    ----------
    model : RandomForestRegressor
    path  : Path, optional
        Output path. Defaults to ``models/sales_forecast_model.pkl``.
    """
    if path is None:
        path = MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"  [INFO] Model saved → {path}")


def load_model(path: Path = None) -> RandomForestRegressor:
    """
    Load a previously saved model from disk.

    Parameters
    ----------
    path : Path, optional
        Path to the .pkl file. Defaults to ``models/sales_forecast_model.pkl``.

    Returns
    -------
    RandomForestRegressor
    """
    if path is None:
        path = MODEL_PATH
    model = joblib.load(path)
    print(f"  [INFO] Model loaded from {path}")
    return model


def feature_importance_report(model: RandomForestRegressor) -> pd.DataFrame:
    """
    Build a sorted feature-importance table from the trained forest.

    Business Reasoning
    ------------------
    Feature importance shows which time patterns drive sales most — e.g. if
    ``Month`` ranks #1, that means seasonal cycles are the dominant driver,
    giving actionable insight for inventory planning.

    Parameters
    ----------
    model : RandomForestRegressor

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [Feature, Importance] sorted descending.
    """
    fi = pd.DataFrame({
        "Feature":    FEATURE_COLS,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    print("\n  ── Feature Importances ─────────────────────────")
    for _, row in fi.iterrows():
        bar = "█" * int(row["Importance"] * 40)
        print(f"  {row['Feature']:20s} {bar}  {row['Importance']:.4f}")
    print("  ────────────────────────────────────────────────")
    return fi


def training_pipeline(daily_df: pd.DataFrame,
                       test_ratio: float = 0.20) -> dict:
    """
    Orchestrate the full training pipeline.

    Parameters
    ----------
    daily_df : pd.DataFrame
        Preprocessed daily DataFrame.
    test_ratio : float
        Validation holdout fraction.

    Returns
    -------
    dict with keys: model, train_df, test_df, predictions, mae, rmse, mape, feature_importance
    """
    print("\n" + "="*60)
    print("  STAGE 2: MODEL TRAINING & EVALUATION")
    print("="*60)

    train_df, test_df = sequential_split(daily_df, test_ratio)
    model             = train_model(train_df)
    metrics           = evaluate_model(model, test_df)
    fi                = feature_importance_report(model)
    save_model(model)

    print(f"\n  ✅ Training complete. MAE=${metrics['mae']:,.2f}  RMSE=${metrics['rmse']:,.2f}")

    return {
        "model":              model,
        "train_df":           train_df,
        "test_df":            test_df,
        "predictions":        metrics["predictions"],
        "mae":                metrics["mae"],
        "rmse":               metrics["rmse"],
        "mape":               metrics["mape"],
        "feature_importance": fi,
    }


if __name__ == "__main__":
    from preprocessing import preprocess_pipeline
    daily = preprocess_pipeline()
    results = training_pipeline(daily)
