"""
train.py
--------
Trains multiple classifiers on CTU-13 network flow data and saves the best
model to models/best_model.pkl.

Usage:
    python train.py                          # uses data/sample_ctu13.csv
    python train.py --data data/ctu13_s10.csv
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
import warnings
warnings.filterwarnings("ignore")

# Project imports
import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils.feature_engineering import load_and_engineer, get_X_y


# ─── Configuration ─────────────────────────────────────────────────────────────
MODELS_DIR  = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

SUPERVISED_MODELS = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=15, class_weight="balanced",
        random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=10, class_weight="balanced", random_state=42
    ),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced", max_iter=500, random_state=42
        )),
    ]),
}


def evaluate_model(name, model, X_test, y_test):
    """Print and return a dict of evaluation metrics."""
    y_pred  = model.predict(X_test)

    # Get probability scores where available
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = y_pred.astype(float)

    pr_auc  = average_precision_score(y_test, y_score)
    roc_auc = roc_auc_score(y_test, y_score)

    print(f"\n{'-'*55}")
    print(f"  Model : {name}")
    print(f"  PR-AUC  : {pr_auc:.4f}   |   ROC-AUC : {roc_auc:.4f}")
    print(f"{'-'*55}")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Botnet"]))

    return {"name": name, "model": model, "pr_auc": pr_auc, "roc_auc": roc_auc,
            "y_pred": y_pred, "y_score": y_score}


def plot_confusion_matrices(results, y_test, out_dir: Path):
    """Save a grid of confusion matrices for all trained models."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        cm = confusion_matrix(y_test, res["y_pred"])
        ConfusionMatrixDisplay(cm, display_labels=["Normal", "Botnet"]).plot(ax=ax, colorbar=False)
        ax.set_title(f"{res['name']}\nPR-AUC={res['pr_auc']:.3f}")

    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrices.png", dpi=150)
    print(f"\n[OK] Confusion matrix saved -> {out_dir / 'confusion_matrices.png'}")


def plot_feature_importance(model, feature_names: list, out_dir: Path):
    """Save a bar chart of feature importances (RF / GB only)."""
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    if not hasattr(clf, "feature_importances_"):
        return

    fi = pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    fi.plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_title("Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importances.png", dpi=150)
    print(f"[OK] Feature importance chart saved -> {out_dir / 'feature_importances.png'}")


def train(data_path: str):
    print(f"\n[1/5] Loading data from: {data_path}")
    df = load_and_engineer(data_path)
    X, y = get_X_y(df)

    print(f"      Total samples  : {len(df):,}")
    print(f"      Botnet samples : {y.sum():,}  ({y.mean()*100:.2f}%)")
    print(f"      Features       : {list(X.columns)}")

    # ── Train/test split (stratified) ─────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    print(f"\n[2/5] Training {len(SUPERVISED_MODELS)} supervised classifiers...")
    results = []
    for name, model in SUPERVISED_MODELS.items():
        print(f"      -> {name}")
        model.fit(X_train, y_train)
        res = evaluate_model(name, model, X_test, y_test)
        results.append(res)

    # Isolation Forest (unsupervised / anomaly detection)
    print("\n[3/5] Training Isolation Forest (unsupervised anomaly detector)...")
    iso = IsolationForest(n_estimators=200, contamination=y_train.mean(), random_state=42)
    iso.fit(X_train[y_train == 0])   # fit only on normal traffic
    iso_pred  = np.where(iso.predict(X_test) == -1, 1, 0)
    iso_score = -iso.decision_function(X_test)
    iso_pr    = average_precision_score(y_test, iso_score)
    iso_roc   = roc_auc_score(y_test, iso_score)
    print(f"  Isolation Forest  PR-AUC={iso_pr:.4f}  ROC-AUC={iso_roc:.4f}")
    print(classification_report(y_test, iso_pred, target_names=["Normal", "Botnet"]))
    results.append({"name": "Isolation Forest (unsupervised)", "model": iso,
                    "pr_auc": iso_pr, "roc_auc": iso_roc,
                    "y_pred": iso_pred, "y_score": iso_score})

    # Pick best model by PR-AUC
    best = max(results, key=lambda r: r["pr_auc"])
    print(f"\n[4/5] Best model: {best['name']}  (PR-AUC={best['pr_auc']:.4f})")

    # Save models and plots
    print("\n[5/5] Saving artefacts...")
    with open(MODELS_DIR / "best_model.pkl", "wb") as f:
        pickle.dump({"model": best["model"], "name": best["name"],
                     "features": list(X.columns)}, f)
    print(f"[OK] Best model saved -> {MODELS_DIR / 'best_model.pkl'}")

    # Save all models
    all_models = {r["name"]: r["model"] for r in results}
    with open(MODELS_DIR / "all_models.pkl", "wb") as f:
        pickle.dump({"models": all_models, "features": list(X.columns)}, f)

    plot_confusion_matrices(results, y_test, MODELS_DIR)
    plot_feature_importance(best["model"], list(X.columns), MODELS_DIR)

    # Summary table
    summary = pd.DataFrame([{k: r[k] for k in ["name", "pr_auc", "roc_auc"]} for r in results])
    summary.to_csv(MODELS_DIR / "model_comparison.csv", index=False)
    print("\n-- Model Comparison Summary ------------------------------------------")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_ctu13.csv",
                        help="Path to binetflow CSV file")
    args = parser.parse_args()
    train(args.data)
