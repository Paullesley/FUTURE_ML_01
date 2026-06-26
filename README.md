# Sales & Demand Forecasting
### Future Interns — Machine Learning Task 1

**Author:** Paul Lesley  
**Dataset:** Superstore Sales (2014–2017) — 9,994 transactions across Furniture, Office Supplies & Technology  
**Model:** RandomForestRegressor  
**Scope:** Daily revenue forecasting with business-actionable insights

---

## 📌 Project Overview

This project was completed as part of the **Future Interns Machine Learning Internship Programme (Task 1)**. The objective is to build an end-to-end sales and demand forecasting prototype using real historical retail data.

The system ingests raw Superstore order-level transactions, engineers time-based features, trains a Random Forest regression model using a chronological (no data-leakage) split, evaluates the model on a held-out validation set, projects 90 days of future daily sales, and produces four professional business visualisations.

---

## 🗂️ Project Structure

```
FUTURE_ML_01/
├── data/
│   └── sales_data.csv              ← Superstore dataset (UTF-8, used by the pipeline)
├── models/
│   └── sales_forecast_model.pkl    ← Saved RandomForest model (auto-generated)
├── notebooks/
│   └── sales_forecasting.ipynb     ← Jupyter Notebook with business insights
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            ← Data loading, cleaning & feature engineering
│   ├── train_model.py              ← Sequential split, training & evaluation
│   ├── forecast.py                 ← 90-day future sales prediction
│   └── visualize.py                ← Four professional business charts
├── outputs/
│   ├── 01_historical_sales_trend.png
│   ├── 02_actual_vs_predicted.png
│   ├── 03_future_forecast.png
│   └── 04_feature_importance.png
├── main.py                         ← Orchestrator (run this!)
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.9 or higher
- pip

### 1. Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Full Pipeline

**Windows (PowerShell):**
```powershell
$env:PYTHONIOENCODING='utf-8'; python main.py
```

**macOS / Linux:**
```bash
python main.py
```

This single command will:
1. ✅ Preprocess and clean `data/sales_data.csv`
2. ✅ Train and evaluate the RandomForestRegressor
3. ✅ Generate a 90-day forward sales forecast
4. ✅ Save four business charts to `outputs/`
5. ✅ Save the trained model to `models/sales_forecast_model.pkl`

### 4. Launch the Jupyter Notebook (Optional)
```bash
jupyter notebook notebooks/sales_forecasting.ipynb
```

---

## 📊 Business Insights & Operational Recommendations

> Based on the Superstore dataset (Jan 2014 – Dec 2017, 9,994 order line-items).

### 🔍 Key Findings

| Insight | Detail |
|---|---|
| **Overall Growth Trend** | Sales increased substantially overall between 2014 and 2017, despite a small decline in 2015. The strongest year was 2017 (~$733K). |
| **Q4 Seasonal Peak** | November–December consistently produce the highest monthly sales. Best single month: **$118,448** (Nov 2017). |
| **Q1 Slow Period** | January–February is the weakest period. Sales drop sharply after the holiday season. |
| **Technology Leads** | Despite having the fewest transactions (1,847), the Technology category generates the highest average revenue per order. |
| **West Region Dominates** | The West region accounts for ~32% of all orders and the highest cumulative sales. |
| **Weekend Patterns** | Weekend transaction volume and average sales were lower than weekday levels, suggesting that weekday activity contributes more strongly to total revenue. |
| **DayOfYear & Days_Since_Start** | The model's top two features — the importance of `Days_Since_Start` suggests the model relies strongly on the long-term time trend when generating predictions. |

### 💡 Actionable Recommendations for Store Managers

#### 📦 Inventory Planning
- **Q4 Ramp-Up:** Consider gradually increasing Technology and Furniture stock before the historically strong Q4 period, while monitoring actual sales velocity before committing to large orders.
- **Q1 Wind-Down:** Reduce new stock orders in January and use the slow period to clear excess furniture inventory with targeted discounts.
- **Office Supplies:** Maintain steady stock year-round; this category is volume-driven and consistently in demand across all months.

#### 👥 Staffing
- **Peak Staffing:** Schedule maximum staff during October–December. The forecast shows the highest projected daily revenues falling in this window.
- **Lean Staffing:** Reduce shift overlap in January–February; the historical dip in Q1 is consistent across all four years in the dataset.

#### 🚀 Marketing & Promotions
- **Q4 Campaign:** Launch promotional campaigns targeting Consumer and Corporate segments in mid-October to capture early holiday spending.
- **Q1 Recovery:** Run discount-led promotions in January (e.g. "New Year Stock Clearance") to convert slow-period inventory into cash flow.
- **West Region Expansion:** Invest in marketing in the West, which already has strong order density — high ROI potential for loyalty programmes.

#### 📉 Risk Mitigation
- The model achieved an **MAE of $1,619.22** and an **RMSE of $2,339.38**. MAPE was unusually high because the dataset includes days with very low sales, so percentage errors become exaggerated. The model is better interpreted as a general trend forecasting prototype rather than a precise daily prediction system. The model performs slightly better than a simple average-sales baseline (R² ≈ 0.14, WAPE ≈ 72.4%), and its daily forecasts are most useful for identifying directional trends and seasonal patterns.

---

## 🤖 Model Details

| Parameter | Value |
|---|---|
| Algorithm | RandomForestRegressor |
| n_estimators | 300 |
| max_depth | 15 |
| min_samples_leaf | 3 |
| Train/Test Split | Sequential 80/20 (time-ordered) |
| Training days | 1,166 (Jan 2014 – Mar 2017) |
| Test days | 292 (Mar 2017 – Dec 2017) |
| MAE | $1,619.22/day |
| RMSE | $2,339.38/day |
| 90-Day Forecast Total | $116,926.76 |
| Average Daily Forecast | $1,299.19 |
| Features | Year, Month, Day, DayOfWeek, DayOfYear, WeekOfYear, Quarter, Is_Weekend, Month_Sin, Month_Cos, Days_Since_Start |
| Target | Daily aggregated Sales (USD) |

---

## 📄 License

This project is submitted as part of the Future Interns Internship Programme and is intended for educational purposes.

