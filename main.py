"""
main.py
=======
Orchestrator script for the Superstore Sales Forecasting pipeline.

Execution order:
  1. Preprocess  - Load, clean, and engineer features from the raw CSV.
  2. Train       - Fit RandomForestRegressor, evaluate, and save the model.
  3. Forecast    - Generate a 90-day forward-looking prediction.
  4. Visualize   - Produce four business-ready PNG charts.

Usage:
    python main.py

Author: Paullesley Allwyn
Project: Future Interns - ML Task 1: Sales & Demand Forecasting
"""

import sys
import io
import time

# Fix Windows console encoding so UTF-8 characters print correctly
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pathlib import Path

# ── Make sure `src/` is on the Python path ───────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from preprocessing import preprocess_pipeline
from train_model   import training_pipeline
from forecast      import generate_forecast
from visualize     import run_all_visualizations


# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(title: str, width: int = 64) -> None:
    """Print a formatted section banner to the terminal."""
    border = "=" * width
    print(f"\n+{border}+")
    print(f"|  {title:<{width - 2}}|")
    print(f"+{border}+")


def _elapsed(start: float) -> str:
    """Return a human-readable elapsed-time string."""
    secs = time.time() - start
    return f"{secs:.1f}s" if secs < 60 else f"{secs / 60:.1f}m"


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    pipeline_start = time.time()

    print("\n" + "#" * 66)
    print("#  FUTURE INTERNS - ML Task 1: Sales & Demand Forecasting      #")
    print("#  Author : Paullesley Allwyn                                   #")
    print("#  Dataset: Superstore (2014-2017)  |  Model: RandomForest      #")
    print("#" * 66)

    # ── Stage 1: Preprocessing ───────────────────────────────────────────────
    _banner("STAGE 1 / 4  ▸  Data Preprocessing")
    t0    = time.time()
    daily = preprocess_pipeline()
    print(f"\n  ⏱  Preprocessing done in {_elapsed(t0)}")

    # ── Stage 2: Training ────────────────────────────────────────────────────
    _banner("STAGE 2 / 4  ▸  Model Training & Evaluation")
    t0      = time.time()
    results = training_pipeline(daily, test_ratio=0.20)
    print(f"\n  ⏱  Training done in {_elapsed(t0)}")

    # ── Stage 3: Forecasting ─────────────────────────────────────────────────
    _banner("STAGE 3 / 4  ▸  90-Day Future Forecast")
    t0          = time.time()
    forecast_df = generate_forecast(daily, horizon_days=90,
                                    model=results["model"])
    print(f"\n  ⏱  Forecasting done in {_elapsed(t0)}")

    # ── Stage 4: Visualizations ──────────────────────────────────────────────
    _banner("STAGE 4 / 4  ▸  Generating Business Visualizations")
    t0 = time.time()
    run_all_visualizations(
        daily_df              = daily,
        test_df               = results["test_df"],
        predictions           = results["predictions"],
        forecast_df           = forecast_df,
        feature_importance_df = results["feature_importance"],
    )
    print(f"\n  ⏱  Visualization done in {_elapsed(t0)}")

    # ── Final Summary ─────────────────────────────────────────────────────────
    _banner("PIPELINE COMPLETE  ▸  Summary")
    print(f"\n  {'Metric':<28} {'Value':>16}")
    print(f"  {'-'*44}")
    print(f"  {'Total training days':<28} {len(results['train_df']):>16,}")
    print(f"  {'Total test days':<28} {len(results['test_df']):>16,}")
    print(f"  {'MAE (daily)':<28} ${results['mae']:>15,.2f}")
    print(f"  {'RMSE (daily)':<28} ${results['rmse']:>15,.2f}")
    print(f"  {'MAPE':<28} {results['mape']:>15.2f}%")
    print(f"  {'90-day forecast total':<28} ${forecast_df['Forecasted_Sales'].sum():>15,.2f}")
    print(f"  {'Average daily forecast':<28} ${forecast_df['Forecasted_Sales'].mean():>15,.2f}")
    print(f"\n  Total pipeline time: {_elapsed(pipeline_start)}")
    print(f"\n  Outputs saved to  : {ROOT / 'outputs'}")
    print(f"  Model saved to    : {ROOT / 'models' / 'sales_forecast_model.pkl'}")
    print(f"\n  Open the notebooks/ folder and run sales_forecasting.ipynb")
    print(f"  for detailed business insights and recommendations.")
    print()


if __name__ == "__main__":
    main()
