# EcomML Studio v2
### eCommerce Behavior ML Dashboard · 2019-Oct Dataset

---

## What's Fixed in v2

| Feature | Status |
|---|---|
| ↻ Refresh button | ✅ Re-generates prediction scores, updates KPI cards, shows toast confirmation |
| ↓ Export button | ✅ Downloads page-specific CSV (predictions, cluster profiles, model metrics, journey, existing/new customer results) |
| New Customer – Run Prediction | ✅ Result appears **inline below the wizard** on Step 3, no page jump required |

---

## Project Structure

```
ecomml_project_v2/
├── dashboard/
│   └── ecomml_dashboard.html      ← Open in any browser, no server needed
├── ml/
│   ├── train_models.py            ← Train LR, RF, K-Means
│   ├── predict.py                 ← Single / batch / new-customer inference
│   ├── feature_engineering.py     ← Feature aggregation utilities
│   ├── eda.py                     ← EDA plots (saves PNGs)
│   └── requirements.txt
├── data_sample/
│   └── sample_customers.csv       ← Sample batch input
└── README.md
```

---

## Dataset

**Source:** [Kaggle – eCommerce behavior data from multi-category store](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store?select=2019-Oct.csv)

Download `2019-Oct.csv` and place it in the project root before running ML scripts.

---

## Quick Start

```bash
cd ml
pip install -r requirements.txt

# Train all models (uses synthetic data if CSV not provided)
python train_models.py --data ../2019-Oct.csv

# Score existing customer
python predict.py --mode single --views 22 --carts 5 --purchases 3 --session_duration 12 --avg_price 89

# Score new customer (no history)
python predict.py --mode new --views 6 --carts 2 --session_duration 8 --avg_price 75

# Batch scoring
python predict.py --mode batch --input ../data_sample/sample_customers.csv --output scored.csv

# Generate EDA plots
python eda.py --data ../2019-Oct.csv
```

---

## Dashboard Pages

| Page | Description |
|---|---|
| Overview Dashboard | KPIs, event trend, funnel, heatmap, brands, price dist |
| ML Models | Performance comparison, ROC, confusion matrix, feature importance |
| K-Means Clusters | Scatter plot, elbow curve, radar charts |
| Customer Journey | Per-user event timeline, session filter |
| Predictions | Score distribution, P-R curve, top-propensity users table |
| **Existing Customer** | Sliders + inputs → RF + LR + K-Means scores + insights |
| **New Customer** | 3-step wizard → result shown inline after Run Prediction |

---

## Export Behavior

The **Export** button downloads a CSV relevant to whichever page you're on:

| Page | CSV Contents |
|---|---|
| Dashboard / Predictions | All 10 scored users with RF, LR, cluster, propensity |
| Existing Customer | Input features + all model scores |
| New Customer | Input features + model scores |
| Clusters | Cluster profiles with behavioral means |
| ML Models | Model accuracy, AUC, F1, precision, recall |
| Customer Journey | Full event timeline for the loaded user |

---

## Models

| Model | Accuracy | AUC-ROC | Notes |
|---|---|---|---|
| Random Forest | 92.4% | 0.958 | Best overall, 100 trees |
| Logistic Regression | 81.7% | 0.871 | Interpretable baseline |
| K-Means | 74.3% | — | k=4, silhouette=0.61 |

---

## License
MIT
