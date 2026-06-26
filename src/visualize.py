"""
src/visualize.py
================
Business-friendly, publication-quality charts for the Superstore sales forecast.

Three charts are generated and saved as PNGs to the ``outputs/`` directory:

  1. Historical Sales Trend       – 30-day rolling average with raw daily bars.
  2. Actual vs. Predicted Sales   – Model accuracy on the validation (test) set.
  3. Future Sales Forecast        – Forward-looking 90-day prediction with
                                    confidence shading.

Design principles:
  • Dark background with high-contrast accent colours (professional look).
  • Annotations for peak periods and key business events.
  • Clear axis labels, grid lines, and legends — readable by a non-technical audience.

Author: Paullesley Allwyn
Project: Future Interns – ML Task 1: Sales & Demand Forecasting
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend (works on headless servers)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Styling constants ────────────────────────────────────────────────────────
BG_COLOR        = "#0D1117"   # GitHub-dark background
PANEL_COLOR     = "#161B22"   # Slightly lighter panel
GRID_COLOR      = "#21262D"   # Subtle grid
TEXT_COLOR      = "#E6EDF3"   # Almost-white text
ACCENT_BLUE     = "#58A6FF"   # Bright blue (historical / actual)
ACCENT_ORANGE   = "#FF7C43"   # Warm orange (predicted)
ACCENT_GREEN    = "#3FB950"   # Vivid green (forecast)
ACCENT_PURPLE   = "#BC8CFF"   # Soft purple (rolling avg / fill)
ACCENT_RED      = "#F85149"   # Alert red (peak annotations)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

plt.rcParams.update({
    "figure.facecolor":  BG_COLOR,
    "axes.facecolor":    PANEL_COLOR,
    "axes.edgecolor":    GRID_COLOR,
    "axes.labelcolor":   TEXT_COLOR,
    "axes.titlecolor":   TEXT_COLOR,
    "axes.titlesize":    14,
    "axes.labelsize":    11,
    "xtick.color":       TEXT_COLOR,
    "ytick.color":       TEXT_COLOR,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "grid.color":        GRID_COLOR,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "legend.facecolor":  PANEL_COLOR,
    "legend.edgecolor":  GRID_COLOR,
    "legend.labelcolor": TEXT_COLOR,
    "legend.fontsize":   9,
    "text.color":        TEXT_COLOR,
    "font.family":       "DejaVu Sans",
})


# ── Helper ────────────────────────────────────────────────────────────────────

def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _add_watermark(fig: plt.Figure) -> None:
    """Add a subtle author watermark in the bottom-right corner."""
    fig.text(0.99, 0.01, "© Paullesley Allwyn | Future Interns – ML Task 1",
             ha="right", va="bottom", fontsize=7,
             color=TEXT_COLOR, alpha=0.4)


def _save(fig: plt.Figure, filename: str) -> None:
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [INFO] Saved → {path}")


# ── Chart 1: Historical Sales Trend ──────────────────────────────────────────

def plot_historical_trend(daily_df: pd.DataFrame) -> None:
    """
    Chart 1 – Historical Daily Sales with a 30-Day Rolling Average.

    Business Insight Shown:
    -----------------------
    * The raw daily bars expose high-frequency noise (e.g. order-delivery timing).
    * The 30-day rolling average reveals the smooth underlying seasonal trend —
      the critical signal for inventory and staffing decisions.
    * Annotated peak months help management identify the strongest selling periods.

    Parameters
    ----------
    daily_df : pd.DataFrame
        Preprocessed daily DataFrame with columns ``Order Date`` and ``Sales``.
    """
    _ensure_output_dir()
    print("  [INFO] Generating Chart 1: Historical Sales Trend …")

    DATE_COL   = "Order Date"
    TARGET_COL = "Sales"

    df = daily_df[[DATE_COL, TARGET_COL]].copy()
    df = df.sort_values(DATE_COL)
    df["Rolling_30"] = df[TARGET_COL].rolling(30, center=True, min_periods=1).mean()
    df["Rolling_7"]  = df[TARGET_COL].rolling(7,  center=True, min_periods=1).mean()

    # Monthly aggregation for bar chart
    df["Month_Period"] = df[DATE_COL].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("Month_Period")[TARGET_COL].sum().reset_index()

    fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                             gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle("Historical Sales Performance  |  Superstore (2014–2017)",
                 fontsize=16, fontweight="bold", y=0.98, color=TEXT_COLOR)

    # ── Top panel: daily + rolling avg ─────────────────────────────────────
    ax1 = axes[0]
    ax1.bar(df[DATE_COL], df[TARGET_COL],
            color=ACCENT_BLUE, alpha=0.20, width=1.0, label="Daily Sales")
    ax1.plot(df[DATE_COL], df["Rolling_30"],
             color=ACCENT_ORANGE, linewidth=2.2, label="30-Day Rolling Avg", zorder=3)
    ax1.plot(df[DATE_COL], df["Rolling_7"],
             color=ACCENT_PURPLE, linewidth=1.0, alpha=0.5, label="7-Day Rolling Avg", zorder=2)

    # Annotate overall peak day
    peak_idx  = df[TARGET_COL].idxmax()
    peak_date = df.loc[peak_idx, DATE_COL]
    peak_val  = df.loc[peak_idx, TARGET_COL]
    ax1.annotate(
        f"Peak: ${peak_val:,.0f}\n{peak_date.strftime('%b %d, %Y')}",
        xy=(peak_date, peak_val),
        xytext=(peak_date - pd.Timedelta(days=180), peak_val * 0.85),
        fontsize=8, color=ACCENT_RED,
        arrowprops=dict(arrowstyle="->", color=ACCENT_RED, lw=1.2),
    )

    ax1.set_ylabel("Daily Sales (USD)", fontweight="bold")
    ax1.set_title("Daily Sales & 30-Day Rolling Average", fontsize=12, pad=10)
    ax1.legend(loc="upper left")
    ax1.grid(True, axis="y")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f"${x:,.0f}"))

    # ── Bottom panel: monthly bar chart ────────────────────────────────────
    ax2 = axes[1]
    colors = [ACCENT_GREEN if v == monthly[TARGET_COL].max() else ACCENT_BLUE
              for v in monthly[TARGET_COL]]
    bars = ax2.bar(monthly["Month_Period"], monthly[TARGET_COL],
                   color=colors, alpha=0.80, width=20)

    # Label the peak month
    peak_m_idx = monthly[TARGET_COL].idxmax()
    ax2.annotate(
        f"Best Month\n${monthly.loc[peak_m_idx, TARGET_COL]:,.0f}",
        xy=(monthly.loc[peak_m_idx, "Month_Period"],
            monthly.loc[peak_m_idx, TARGET_COL]),
        xytext=(0, 8), textcoords="offset points",
        ha="center", fontsize=7.5, color=ACCENT_GREEN, fontweight="bold",
    )

    ax2.set_ylabel("Monthly Sales (USD)", fontweight="bold")
    ax2.set_title("Monthly Aggregated Sales", fontsize=12, pad=10)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f"${x:,.0f}"))
    ax2.grid(True, axis="y")

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    _add_watermark(fig)
    _save(fig, "01_historical_sales_trend.png")


# ── Chart 2: Actual vs. Predicted ────────────────────────────────────────────

def plot_actual_vs_predicted(test_df: pd.DataFrame,
                              predictions: np.ndarray) -> None:
    """
    Chart 2 – Actual vs. Predicted Daily Sales on the validation set.

    Business Insight Shown:
    -----------------------
    * Overlaying the model's predictions on real data gives a store manager
      confidence that the model has learned genuine seasonal patterns.
    * The residual (error) subplot reveals systematic over/under-prediction
      in specific periods — e.g. the model may consistently under-predict
      during the Q4 holiday surge.

    Parameters
    ----------
    test_df     : pd.DataFrame  – held-out test portion of the daily DataFrame.
    predictions : np.ndarray    – model predictions aligned with test_df rows.
    """
    _ensure_output_dir()
    print("  [INFO] Generating Chart 2: Actual vs Predicted …")

    DATE_COL   = "Order Date"
    TARGET_COL = "Sales"

    df = test_df[[DATE_COL, TARGET_COL]].copy().reset_index(drop=True)
    df["Predicted"] = predictions
    df["Residual"]  = df[TARGET_COL] - df["Predicted"]

    # Metrics for annotation
    mae  = np.mean(np.abs(df["Residual"]))
    rmse = np.sqrt(np.mean(df["Residual"] ** 2))

    fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                             gridspec_kw={"height_ratios": [2.5, 1]})
    fig.suptitle("Model Validation: Actual vs. Predicted Sales",
                 fontsize=16, fontweight="bold", y=0.98, color=TEXT_COLOR)

    # ── Top: Actual vs Predicted lines ─────────────────────────────────────
    ax1 = axes[0]
    ax1.fill_between(df[DATE_COL], df[TARGET_COL], alpha=0.15,
                     color=ACCENT_BLUE, label="_nolegend_")
    ax1.plot(df[DATE_COL], df[TARGET_COL],
             color=ACCENT_BLUE, linewidth=1.8, label="Actual Sales", alpha=0.9)
    ax1.plot(df[DATE_COL], df["Predicted"],
             color=ACCENT_ORANGE, linewidth=1.8, linestyle="--",
             label="Predicted Sales", alpha=0.9)

    # Metrics box
    metric_text = f"MAE:  ${mae:,.0f}\nRMSE: ${rmse:,.0f}"
    props = dict(boxstyle="round,pad=0.5", facecolor=PANEL_COLOR,
                 edgecolor=ACCENT_ORANGE, alpha=0.9)
    ax1.text(0.02, 0.95, metric_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment="top", bbox=props,
             color=TEXT_COLOR, fontfamily="monospace")

    ax1.set_ylabel("Daily Sales (USD)", fontweight="bold")
    ax1.set_title("Validation Set  |  Last 20% of Data (Chronological Hold-out)", fontsize=12, pad=10)
    ax1.legend(loc="upper right")
    ax1.grid(True, axis="y")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f"${x:,.0f}"))

    # ── Bottom: Residuals ──────────────────────────────────────────────────
    ax2 = axes[1]
    pos_mask = df["Residual"] >= 0
    ax2.bar(df.loc[pos_mask, DATE_COL], df.loc[pos_mask, "Residual"],
            color=ACCENT_GREEN, alpha=0.7, width=1.0, label="Under-predicted")
    ax2.bar(df.loc[~pos_mask, DATE_COL], df.loc[~pos_mask, "Residual"],
            color=ACCENT_RED, alpha=0.7, width=1.0, label="Over-predicted")
    ax2.axhline(0, color=TEXT_COLOR, linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Residual (USD)", fontweight="bold")
    ax2.set_title("Prediction Residuals (Actual − Predicted)", fontsize=12, pad=10)
    ax2.legend(loc="upper right")
    ax2.grid(True, axis="y")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f"${x:+,.0f}"))

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    _add_watermark(fig)
    _save(fig, "02_actual_vs_predicted.png")


# ── Chart 3: Future Forecast ──────────────────────────────────────────────────

def plot_future_forecast(daily_df: pd.DataFrame,
                          forecast_df: pd.DataFrame) -> None:
    """
    Chart 3 – 90-Day Future Sales Forecast.

    Business Insight Shown:
    -----------------------
    * The final N historical days are shown as context so managers can
      intuitively see the forecast continuing naturally from known data.
    * Uncertainty shading (±15 % of predicted values) communicates that
      forecasts are estimates, not certainties — promoting responsible planning.
    * Weekend highlights help staffing managers see when peak days are forecast.

    Parameters
    ----------
    daily_df    : pd.DataFrame  – full preprocessed daily history.
    forecast_df : pd.DataFrame  – output of generate_forecast().
    """
    _ensure_output_dir()
    print("  [INFO] Generating Chart 3: Future Sales Forecast …")

    DATE_COL = "Order Date"
    FC_COL   = "Forecasted_Sales"

    # Show last 120 days of history as context
    history = daily_df[[DATE_COL, "Sales"]].copy().tail(120)
    fc      = forecast_df[[DATE_COL, FC_COL]].copy()

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.suptitle("90-Day Sales Forecast  |  Future Revenue Projection",
                 fontsize=16, fontweight="bold", y=0.98, color=TEXT_COLOR)

    # Historical context
    ax.fill_between(history[DATE_COL], history["Sales"],
                    color=ACCENT_BLUE, alpha=0.12)
    ax.plot(history[DATE_COL], history["Sales"],
            color=ACCENT_BLUE, linewidth=1.5, label="Historical Sales (last 120 days)", alpha=0.8)

    # Forecast line
    ax.plot(fc[DATE_COL], fc[FC_COL],
            color=ACCENT_GREEN, linewidth=2.5, label="90-Day Forecast", zorder=5)

    # Uncertainty band ±15 %
    upper = fc[FC_COL] * 1.15
    lower = fc[FC_COL] * 0.85
    ax.fill_between(fc[DATE_COL], lower, upper,
                    color=ACCENT_GREEN, alpha=0.15, label="±15% Planning Range")

    # Vertical separator
    split_date = history[DATE_COL].max()
    ax.axvline(split_date, color=TEXT_COLOR, linewidth=1.2,
               linestyle=":", alpha=0.6, label="Forecast Start")
    ax.text(split_date + pd.Timedelta(days=2),
            ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else fc[FC_COL].max() * 0.95,
            "  Forecast →", color=ACCENT_GREEN, fontsize=9, alpha=0.8)

    # Shade weekend days in forecast period
    for _, row in fc.iterrows():
        if row[DATE_COL].dayofweek >= 5:   # Saturday / Sunday
            ax.axvspan(row[DATE_COL] - pd.Timedelta(hours=12),
                       row[DATE_COL] + pd.Timedelta(hours=12),
                       color=ACCENT_PURPLE, alpha=0.05)

    # Annotate forecast peak
    peak_idx      = fc[FC_COL].idxmax()
    peak_fc_date  = fc.loc[peak_idx, DATE_COL]
    peak_fc_val   = fc.loc[peak_idx, FC_COL]
    ax.annotate(
        f"Forecast Peak\n${peak_fc_val:,.0f}",
        xy=(peak_fc_date, peak_fc_val),
        xytext=(peak_fc_date - pd.Timedelta(days=20), peak_fc_val * 1.10),
        fontsize=8.5, color=ACCENT_RED, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=ACCENT_RED, lw=1.2),
    )

    # Summary stats box
    total_fc = fc[FC_COL].sum()
    avg_fc   = fc[FC_COL].mean()
    stats_text = (f"90-Day Forecast\n"
                  f"Total : ${total_fc:,.0f}\n"
                  f"Daily Avg: ${avg_fc:,.0f}")
    props = dict(boxstyle="round,pad=0.5", facecolor=PANEL_COLOR,
                 edgecolor=ACCENT_GREEN, alpha=0.9)
    ax.text(0.015, 0.97, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top", bbox=props,
            color=TEXT_COLOR, fontfamily="monospace")

    ax.set_xlabel("Date", fontweight="bold")
    ax.set_ylabel("Daily Sales (USD)", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, axis="y")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%Y"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f"${x:,.0f}"))

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    _add_watermark(fig)
    _save(fig, "03_future_forecast.png")


# ── Chart 4 (Bonus): Feature Importance ──────────────────────────────────────

def plot_feature_importance(feature_importance_df: pd.DataFrame) -> None:
    """
    Bonus Chart – Horizontal bar chart of model feature importances.

    Business Insight Shown:
    -----------------------
    Which time-feature matters most?  If ``Days_Since_Start`` ranks highest,
    long-term growth is the primary driver.  If ``Month`` ranks highest,
    seasonal cycles dominate — both shape very different business strategies.

    Parameters
    ----------
    feature_importance_df : pd.DataFrame
        Output of ``feature_importance_report()`` with columns [Feature, Importance].
    """
    _ensure_output_dir()
    print("  [INFO] Generating Bonus Chart: Feature Importance …")

    fi = feature_importance_df.sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("What Drives Sales?  |  Model Feature Importance",
                 fontsize=16, fontweight="bold", y=0.98, color=TEXT_COLOR)

    colors_bar = [ACCENT_GREEN if v == fi["Importance"].max() else ACCENT_BLUE
                  for v in fi["Importance"]]
    bars = ax.barh(fi["Feature"], fi["Importance"],
                   color=colors_bar, edgecolor=PANEL_COLOR, height=0.6)

    for bar, val in zip(bars, fi["Importance"]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left",
                fontsize=9, color=TEXT_COLOR)

    ax.set_xlabel("Relative Importance Score", fontweight="bold")
    ax.set_title("RandomForest Feature Importance (Higher = More Predictive)",
                 fontsize=11, pad=10)
    ax.grid(True, axis="x")
    ax.set_xlim(0, fi["Importance"].max() * 1.18)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    _add_watermark(fig)
    _save(fig, "04_feature_importance.png")


# ── Main visualisation orchestrator ──────────────────────────────────────────

def run_all_visualizations(daily_df: pd.DataFrame,
                            test_df: pd.DataFrame,
                            predictions: np.ndarray,
                            forecast_df: pd.DataFrame,
                            feature_importance_df: pd.DataFrame) -> None:
    """
    Generate all four charts in sequence.

    Parameters
    ----------
    daily_df              : full preprocessed daily DataFrame
    test_df               : validation/test portion
    predictions           : model predictions on the test set
    forecast_df           : future forecast DataFrame
    feature_importance_df : feature importance table
    """
    print("\n" + "="*60)
    print("  STAGE 4: GENERATING VISUALIZATIONS")
    print("="*60)

    plot_historical_trend(daily_df)
    plot_actual_vs_predicted(test_df, predictions)
    plot_future_forecast(daily_df, forecast_df)
    plot_feature_importance(feature_importance_df)

    print(f"\n  ✅ All charts saved to: {OUTPUT_DIR}")


import matplotlib.ticker   # needed for FuncFormatter used in helpers above


if __name__ == "__main__":
    from preprocessing import preprocess_pipeline
    from train_model   import training_pipeline
    from forecast      import generate_forecast

    daily   = preprocess_pipeline()
    results = training_pipeline(daily)
    fc      = generate_forecast(daily, horizon_days=90, model=results["model"])

    run_all_visualizations(
        daily_df              = daily,
        test_df               = results["test_df"],
        predictions           = results["predictions"],
        forecast_df           = fc,
        feature_importance_df = results["feature_importance"],
    )
