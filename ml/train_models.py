"""
EcomML – Model Training Pipeline
Dataset : eCommerce behavior data from multi-category store (2019-Oct.csv)
Source  : https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store

Models trained
--------------
1. Logistic Regression  (sklearn)
2. Random Forest        (sklearn)
3. K-Means Clustering   (sklearn)

Run
---
    pip install -r requirements.txt
    python train_models.py --data 2019-Oct.csv

Output
------
    models/logistic_regression.pkl
    models/random_forest.pkl
    models/kmeans.pkl
    models/scaler.pkl
    models/feature_columns.json
    reports/metrics_report.json
    reports/classification_report_rf.txt
    reports/classification_report_lr.txt
"""

import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")

# pyrefly: ignore[missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore[missing-import]
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, confusion_matrix, f1_score,
    precision_score, recall_score
)

# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────
MODELS_DIR  = "models"
REPORTS_DIR = "reports"
RANDOM_SEED = 42
TEST_SIZE   = 0.20
N_CLUSTERS  = 4

# ──────────────────────────────────────────────
#  FEATURE ENGINEERING
# ──────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw event-level rows into one row per user_session.

    Raw columns expected (from 2019-Oct.csv):
        event_time, event_type, product_id, category_id,
        category_code, brand, price, user_id, user_session

    Engineered features:
        view_count        – number of 'view' events
        cart_count        – number of 'cart' events
        purchase_count    – number of 'purchase' events  (TARGET)
        session_duration  – max-min event_time in minutes
        avg_price_viewed  – mean price across all viewed products
        unique_categories – distinct category_code values seen
        unique_brands     – distinct brands interacted with
        brand_loyalty     – max brand frequency / total events (0–1)
        cart_to_view_ratio – cart_count / (view_count + 1)
        price_range       – max price – min price of products seen
        label             – 1 if user purchased in this session else 0
    """
    print("[feature_eng] Parsing event_time …")
    df = df.copy()
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["event_time"])

    print("[feature_eng] Creating event flags …")
    df["is_view"] = (df["event_type"] == "view").astype(int)
    df["is_cart"] = (df["event_type"] == "cart").astype(int)
    df["is_purchase"] = (df["event_type"] == "purchase").astype(int)

    print("[feature_eng] Aggregating per user_session …")
    # Vectorized session-level aggregation
    agg_df = df.groupby(["user_id", "user_session"]).agg(
        view_count=("is_view", "sum"),
        cart_count=("is_cart", "sum"),
        purchase_count=("is_purchase", "sum"),
        min_time=("event_time", "min"),
        max_time=("event_time", "max"),
        avg_price_viewed=("price", "mean"),
        min_price=("price", "min"),
        max_price=("price", "max"),
        unique_categories=("category_code", "nunique"),
        unique_brands=("brand", "nunique")
    ).reset_index()

    agg_df["session_duration"] = (agg_df["max_time"] - agg_df["min_time"]).dt.total_seconds() / 60.0
    agg_df["avg_price_viewed"] = agg_df["avg_price_viewed"].fillna(0)
    agg_df["price_range"] = (agg_df["max_price"] - agg_df["min_price"]).fillna(0)

    # Drop temporary columns used in aggregation
    feat = agg_df.drop(columns=["min_time", "max_time", "min_price", "max_price"])

    print("[feature_eng] Calculating brand loyalty vectorially …")
    brand_df = df.dropna(subset=["brand"])
    if not brand_df.empty:
        brand_counts = brand_df.groupby(["user_id", "user_session", "brand"]).size().reset_index(name="brand_count")
        max_brand_counts = brand_counts.groupby(["user_id", "user_session"])["brand_count"].max()
        total_brand_counts = brand_counts.groupby(["user_id", "user_session"])["brand_count"].sum()
        brand_loyalty = (max_brand_counts / total_brand_counts).fillna(0).reset_index(name="brand_loyalty")
        feat = pd.merge(feat, brand_loyalty, on=["user_id", "user_session"], how="left")
        feat["brand_loyalty"] = feat["brand_loyalty"].fillna(0)
    else:
        feat["brand_loyalty"] = 0.0

    feat["cart_to_view_ratio"] = feat["cart_count"] / (feat["view_count"] + 1)
    feat["label"] = (feat["purchase_count"] > 0).astype(int)

    print(f"[feature_eng] Shape after aggregation: {feat.shape}")
    print(f"[feature_eng] Purchase rate: {feat['label'].mean()*100:.2f}%")
    return feat


FEATURE_COLS = [
    "view_count", "cart_count", "session_duration",
    "avg_price_viewed", "unique_categories", "unique_brands",
    "brand_loyalty", "cart_to_view_ratio", "price_range"
]

# ──────────────────────────────────────────────
#  GENERATE SYNTHETIC DATA (used when CSV absent)
# ──────────────────────────────────────────────
def generate_synthetic_data(n: int = 100_000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Creates a realistic synthetic dataset that mirrors the 2019-Oct.csv
    feature distributions for demo / CI runs.
    """
    print(f"[synthetic] Generating {n:,} synthetic sessions …")
    rng = np.random.default_rng(seed)

    view_count        = rng.negative_binomial(5, 0.25, n).clip(0, 100).astype(float)
    cart_count        = (rng.negative_binomial(2, 0.5, n) * (view_count > 0)).clip(0, 30).astype(float)
    purchase_count    = (rng.negative_binomial(1, 0.85, n) * (cart_count > 0)).clip(0, 10).astype(float)
    session_duration  = rng.exponential(8, n).clip(0.1, 120)
    avg_price_viewed  = rng.lognormal(4.2, 1.0, n).clip(1, 2000)
    unique_categories = rng.integers(1, 9, n).astype(float)
    unique_brands     = rng.integers(1, 8, n).astype(float)
    brand_loyalty     = rng.beta(2, 5, n)
    cart_to_view_ratio= cart_count / (view_count + 1)
    price_range       = (avg_price_viewed * rng.uniform(0, 0.8, n)).clip(0)

    label = (purchase_count > 0).astype(int)

    df = pd.DataFrame({
        "view_count": view_count,
        "cart_count": cart_count,
        "session_duration": session_duration,
        "avg_price_viewed": avg_price_viewed,
        "unique_categories": unique_categories,
        "unique_brands": unique_brands,
        "brand_loyalty": brand_loyalty,
        "cart_to_view_ratio": cart_to_view_ratio,
        "price_range": price_range,
        "label": label,
    })
    print(f"[synthetic] Purchase rate: {label.mean()*100:.2f}%")
    return df


# ──────────────────────────────────────────────
#  TRAINING
# ──────────────────────────────────────────────
def train_logistic_regression(X_train, y_train):
    print("[LR] Training Logistic Regression …")
    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    print("[RF] Training Random Forest …")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)
    return model


def train_kmeans(X_scaled):
    print(f"[KM] Training K-Means with k={N_CLUSTERS} …")
    model = KMeans(
        n_clusters=N_CLUSTERS,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=RANDOM_SEED,
    )
    model.fit(X_scaled)
    return model


# ──────────────────────────────────────────────
#  EVALUATION
# ──────────────────────────────────────────────
def evaluate_classifier(name, model, X_test, y_test, scaler=None):
    X_in = scaler.transform(X_test) if scaler else X_test
    y_pred  = model.predict(X_in)
    y_proba = model.predict_proba(X_in)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, zero_division=0)

    metrics = {
        "model": name,
        "accuracy":  round(acc,  4),
        "auc_roc":   round(auc,  4),
        "f1_score":  round(f1,   4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "confusion_matrix": cm,
    }

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"\n{report}")
    return metrics, report


def evaluate_kmeans(model, X_scaled):
    from sklearn.metrics import silhouette_score
    labels   = model.labels_
    inertia  = model.inertia_
    sil      = silhouette_score(X_scaled, labels, sample_size=10_000, random_state=RANDOM_SEED)
    counts   = pd.Series(labels).value_counts().sort_index().to_dict()

    print(f"\n{'='*50}")
    print(f"  K-Means (k={N_CLUSTERS})")
    print(f"{'='*50}")
    print(f"  Inertia         : {inertia:,.1f}")
    print(f"  Silhouette Score: {sil:.4f}")
    for k, v in counts.items():
        print(f"  Cluster {k}       : {v:,} samples ({v/len(labels)*100:.1f}%)")

    return {
        "model":            "K-Means",
        "k":                N_CLUSTERS,
        "inertia":          round(inertia, 2),
        "silhouette_score": round(sil, 4),
        "cluster_sizes":    {str(k): int(v) for k, v in counts.items()},
    }


# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────
def main(data_path: str | None, limit: int = 1000000):
    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # ── Load / generate data ─────────────────
    read_kwargs = {}
    if limit > 0:
        read_kwargs["nrows"] = limit
        print(f"[data] Row limit set to {limit:,}")

    if data_path and os.path.exists(data_path):
        print(f"[data] Loading {data_path} …")
        raw = pd.read_csv(data_path, **read_kwargs)
        print(f"[data] Raw shape: {raw.shape}")
        df = engineer_features(raw)
    else:
        # Load both datasets from data_sample
        data_paths = [
            "../data_sample/2019-Oct.csv",
            "../data_sample/2019-Nov.csv"
        ]
        raw_dfs = []
        for path in data_paths:
            if os.path.exists(path):
                print(f"[data] Loading {path} …")
                df_temp = pd.read_csv(path, **read_kwargs)
                raw_dfs.append(df_temp)
                print(f"[data] Shape: {df_temp.shape}")
        if raw_dfs:
            raw = pd.concat(raw_dfs, ignore_index=True)
            print(f"[data] Combined raw shape: {raw.shape}")
            df = engineer_features(raw)
        else:
            print("[data] No CSVs found – using synthetic data for demo.")
            df = generate_synthetic_data(n=100_000)

    X = df[FEATURE_COLS].values
    y = df["label"].values

    # ── Scale ────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Split ────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"\n[split] Train: {len(X_tr):,}  |  Test: {len(X_te):,}")

    # ── Train ────────────────────────────────
    lr  = train_logistic_regression(X_tr, y_tr)
    rf  = train_random_forest(X_tr, y_tr)
    km  = train_kmeans(X_scaled)

    # ── Evaluate ─────────────────────────────
    lr_metrics, lr_report = evaluate_classifier("Logistic Regression", lr, X_te, y_te)
    rf_metrics, rf_report = evaluate_classifier("Random Forest",        rf, X_te, y_te)
    km_metrics            = evaluate_kmeans(km, X_scaled)

    # ── Save models ──────────────────────────
    joblib.dump(lr,     os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    joblib.dump(rf,     os.path.join(MODELS_DIR, "random_forest.pkl"))
    joblib.dump(km,     os.path.join(MODELS_DIR, "kmeans.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    with open(os.path.join(MODELS_DIR, "feature_columns.json"), "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)

    # ── Save reports ─────────────────────────
    report_data = {
        "logistic_regression": lr_metrics,
        "random_forest":       rf_metrics,
        "kmeans":              km_metrics,
    }
    with open(os.path.join(REPORTS_DIR, "metrics_report.json"), "w") as f:
        json.dump(report_data, f, indent=2)

    with open(os.path.join(REPORTS_DIR, "classification_report_lr.txt"), "w") as f:
        f.write(lr_report)

    with open(os.path.join(REPORTS_DIR, "classification_report_rf.txt"), "w") as f:
        f.write(rf_report)

    print("\n[OK] All models saved to ./models/")
    print("[OK] Reports saved to ./reports/")
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ecomml models")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to 2019-Oct.csv (optional)")
    parser.add_argument("--limit", type=int, default=1000000,
                        help="Limit number of rows read from CSV (default: 1,000,000; set -1 for all)")
    args = parser.parse_args()
    main(args.data, args.limit)
