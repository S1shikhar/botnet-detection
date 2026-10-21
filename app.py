"""
app.py  —  Streamlit Dashboard for Botnet Detection
-----------------------------------------------------
Run with:  streamlit run app.py
"""

import pickle
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, roc_curve, average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score,
)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils.feature_engineering import load_and_engineer, get_X_y, FEATURE_COLS

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Botnet Detection System",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS_DIR = Path("models")
DATA_DIR   = Path("data")

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    path = MODELS_DIR / "all_models.pkl"
    if not path.exists():
        return None, None
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["models"], data["features"]


@st.cache_data
def load_data(path: str):
    df = load_and_engineer(path)
    return df


def color_label(val):
    return "background-color: #FFDCDC;" if val == 1 else "background-color: #D4EDDA;"


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Octicons-package.svg/120px-Octicons-package.svg.png",
    width=60,
)
st.sidebar.title("🕵️ Botnet Detector")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "🔍 Predict on File", "🔬 Model Comparison",
     "🔴 Live Capture", "📄 Export Report", "📖 About"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Page: Dashboard
# ─────────────────────────────────────────────────────────────────────────────
if page == "📊 Dashboard":
    st.title("📊 Botnet Detection — Dashboard")
    st.markdown("Upload or select a CTU-13 binetflow `.csv` to analyse network flows.")

    data_files = list(DATA_DIR.glob("*.csv"))
    col1, col2 = st.columns([3, 1])

    with col1:
        uploaded = st.file_uploader("Upload a binetflow CSV", type=["csv"])

    # Load data
    df = None
    if uploaded:
        tmp_path = DATA_DIR / uploaded.name
        tmp_path.write_bytes(uploaded.read())
        df = load_data(str(tmp_path))
        st.success(f"Loaded `{uploaded.name}` — {len(df):,} flows")
    elif data_files:
        selected = st.selectbox("Or pick an existing file:", data_files, format_func=lambda p: p.name)
        df = load_data(str(selected))

    if df is None:
        st.info("⬆️  Upload a CTU-13 binetflow CSV, or run `python train.py` first to generate sample data.")
        st.stop()

    # ── KPI metrics ────────────────────────────────────────────────────────
    total     = len(df)
    n_botnet  = int(df["is_botnet"].sum())
    n_normal  = total - n_botnet
    pct_bot   = n_botnet / total * 100

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Flows",   f"{total:,}")
    k2.metric("🟢 Normal",     f"{n_normal:,}")
    k3.metric("🔴 Botnet",     f"{n_botnet:,}")
    k4.metric("Infection Rate", f"{pct_bot:.2f}%")

    st.markdown("---")

    # ── Charts ─────────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Traffic Composition")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie([n_normal, n_botnet], labels=["Normal", "Botnet"],
               colors=["#4CAF50", "#F44336"], autopct="%1.1f%%", startangle=90)
        st.pyplot(fig)

    with c2:
        st.subheader("Bytes Distribution (log scale)")
        fig, ax = plt.subplots(figsize=(5, 4))
        for label, color, name in [(0, "#4CAF50", "Normal"), (1, "#F44336", "Botnet")]:
            vals = df[df["is_botnet"] == label]["TotBytes"].clip(upper=1e7)
            ax.hist(np.log1p(vals), bins=50, alpha=0.6, color=color, label=name)
        ax.set_xlabel("log(1 + TotBytes)")
        ax.set_ylabel("Count")
        ax.legend()
        st.pyplot(fig)

    # ── Feature table ───────────────────────────────────────────────────────
    st.subheader("Sample Flows (50 rows)")
    show_cols = ["SrcAddr", "DstAddr", "Proto", "TotPkts", "TotBytes",
                 "Dur", "bytes_per_pkt", "is_botnet"]
    show_cols = [c for c in show_cols if c in df.columns]
    sample = df[show_cols].sample(min(50, len(df))).reset_index(drop=True)
    st.dataframe(
        sample.style.apply(
            lambda row: ["background-color: #FFDCDC" if row["is_botnet"] == 1
                         else "background-color: #D4EDDA"] * len(row),
            axis=1
        ),
        use_container_width=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page: Predict on File
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Predict on File":
    st.title("🔍 Run Detection on Network Flows")

    models, feature_names = load_models()
    if models is None:
        st.warning("No trained models found. Run `python train.py` first.")
        st.stop()

    model_name = st.selectbox("Choose Model", list(models.keys()))
    model = models[model_name]

    uploaded = st.file_uploader("Upload binetflow CSV for prediction", type=["csv"])
    if uploaded is None:
        st.info("Upload a CTU-13 binetflow CSV to run detection.")
        st.stop()

    raw = uploaded.read()
    df  = load_and_engineer(io.StringIO(raw.decode("utf-8")))
    X, y_true = get_X_y(df)

    # Align features
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_names]

    # Predict
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
        y_pred  = (y_score >= 0.5).astype(int)
    else:
        y_pred  = np.where(model.predict(X) == -1, 1, 0)
        y_score = -model.decision_function(X)

    df["predicted_label"] = y_pred
    df["anomaly_score"]   = np.round(y_score, 4)

    st.markdown("---")
    detected = int(y_pred.sum())
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Flows",    f"{len(df):,}")
    k2.metric("🔴 Detected Botnet", f"{detected:,}")
    k3.metric("Detection Rate",  f"{detected/len(df)*100:.2f}%")

    st.markdown("---")

    # Metrics (if labels available)
    if y_true.sum() > 0:
        pr_auc  = average_precision_score(y_true, y_score)
        roc_auc = roc_auc_score(y_true, y_score)
        st.subheader("Evaluation Metrics (ground truth labels found)")
        m1, m2 = st.columns(2)
        m1.metric("PR-AUC",  f"{pr_auc:.4f}")
        m2.metric("ROC-AUC", f"{roc_auc:.4f}")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        # Precision-Recall
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        axes[0].plot(rec, prec, color="#4C72B0", lw=2, label=f"PR-AUC={pr_auc:.3f}")
        axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
        axes[0].set_title("Precision-Recall Curve"); axes[0].legend()
        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        axes[1].plot(fpr, tpr, color="#C44E52", lw=2, label=f"ROC-AUC={roc_auc:.3f}")
        axes[1].plot([0,1],[0,1], "k--", lw=1)
        axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
        axes[1].set_title("ROC Curve"); axes[1].legend()
        st.pyplot(fig)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ConfusionMatrixDisplay(cm, display_labels=["Normal","Botnet"]).plot(ax=ax2, colorbar=False)
        ax2.set_title("Confusion Matrix")
        st.pyplot(fig2)

    # Results table
    st.subheader("Detection Results")
    show = df[["SrcAddr","DstAddr","TotBytes","Dur","predicted_label","anomaly_score"]].copy()
    st.dataframe(
        show.style.apply(
            lambda row: ["background-color: #FFDCDC" if row["predicted_label"] == 1
                         else "background-color: #D4EDDA"] * len(row),
            axis=1
        ),
        use_container_width=True,
    )

    # Download
    csv_bytes = df.to_csv(index=False).encode()
    st.download_button("⬇️ Download Results CSV", csv_bytes, "detection_results.csv", "text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# Page: Model Comparison
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔬 Model Comparison":
    st.title("🔬 Model Comparison")

    csv_path = MODELS_DIR / "model_comparison.csv"
    if csv_path.exists():
        summary = pd.read_csv(csv_path)
        st.dataframe(
            summary.style.highlight_max(subset=["pr_auc","roc_auc"], color="#D4EDDA"),
            use_container_width=True,
        )
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(summary))
        w = 0.35
        ax.bar(x - w/2, summary["pr_auc"],  w, label="PR-AUC",  color="#4C72B0")
        ax.bar(x + w/2, summary["roc_auc"], w, label="ROC-AUC", color="#C44E52")
        ax.set_xticks(x); ax.set_xticklabels(summary["name"], rotation=15, ha="right")
        ax.set_ylim(0, 1.05); ax.legend(); ax.set_title("Model Comparison")
        st.pyplot(fig)
    else:
        st.warning("No comparison data yet. Run `python train.py` first.")

    cm_img = MODELS_DIR / "confusion_matrices.png"
    fi_img = MODELS_DIR / "feature_importances.png"
    if cm_img.exists():
        st.subheader("Confusion Matrices")
        st.image(str(cm_img))
    if fi_img.exists():
        st.subheader("Feature Importances (Best Model)")
        st.image(str(fi_img))



# ─────────────────────────────────────────────────────────────────────────────
# Page: Live Capture
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔴 Live Capture":
    st.title("🔴 Live Network Capture & Detection")

    models, feature_names = load_models()
    if models is None:
        st.warning("No trained models found. Run `python train.py` first.")
        st.stop()

    # ── Npcap check ───────────────────────────────────────────────────────────
    st.info("""
**Requirement:** Live capture needs [Npcap](https://npcap.com/#download) installed on Windows.
If you see a Scapy/socket error, install Npcap first (free, takes 1 minute).
""")

    model_name = st.selectbox("Detection Model", list(models.keys()))
    model = models[model_name]
    n_packets = st.slider("Packets to capture", min_value=20, max_value=500, value=100, step=20)
    iface_input = st.text_input("Network interface (leave blank for default)", value="")

    col1, col2 = st.columns([1, 3])
    start = col1.button("Start Capture", type="primary")
    stop_ph = col2.empty()

    if start:
        try:
            from scapy.all import sniff, IP, TCP, UDP
        except ImportError:
            st.error("Scapy not installed. Run: pip install scapy")
            st.stop()

        progress = st.progress(0, text="Capturing packets...")
        status   = st.empty()

        # ── Capture packets ────────────────────────────────────────────────
        captured = []

        def process_pkt(pkt):
            if IP in pkt:
                proto = 0 if TCP in pkt else (1 if UDP in pkt else 2)
                sport = int(pkt.sport) if (TCP in pkt or UDP in pkt) else 0
                dport = int(pkt.dport) if (TCP in pkt or UDP in pkt) else 0
                size  = len(pkt)
                captured.append({
                    "SrcAddr": pkt[IP].src, "DstAddr": pkt[IP].dst,
                    "Proto": proto, "TotPkts": 1,
                    "TotBytes": size, "SrcBytes": size,
                    "sport_numeric": sport, "dport_numeric": dport,
                    "Dur": 0.1, "State": 0,
                    "bytes_per_pkt": float(size),
                    "src_bytes_ratio": 1.0, "dst_bytes": 0.0,
                })
            progress.progress(min(len(captured) / n_packets, 1.0),
                               text=f"Captured {len(captured)}/{n_packets} packets...")

        try:
            kwargs = {"prn": process_pkt, "count": n_packets, "store": False}
            if iface_input.strip():
                kwargs["iface"] = iface_input.strip()
            sniff(**kwargs)
        except Exception as e:
            st.error(f"Capture error: {e}\n\nMake sure Npcap is installed and run as Administrator.")
            st.stop()

        progress.empty()
        if not captured:
            st.warning("No IP packets captured. Check your interface.")
            st.stop()

        # ── Run detection ──────────────────────────────────────────────────
        live_df = pd.DataFrame(captured)
        X_live  = live_df[[c for c in feature_names if c in live_df.columns]].copy()
        for col in feature_names:
            if col not in X_live.columns:
                X_live[col] = 0
        X_live = X_live[feature_names].fillna(0)

        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_live)[:, 1]
            preds  = (scores >= 0.5).astype(int)
        else:
            preds  = np.where(model.predict(X_live) == -1, 1, 0)
            scores = -model.decision_function(X_live)

        live_df["predicted_label"] = preds
        live_df["anomaly_score"]   = np.round(scores, 4)

        # ── Summary KPIs ───────────────────────────────────────────────────
        n_det = int(preds.sum())
        st.success(f"Capture complete! Analysed {len(live_df)} packets.")
        k1, k2, k3 = st.columns(3)
        k1.metric("Total Packets",  f"{len(live_df):,}")
        k2.metric("Botnet Detected", f"{n_det:,}", delta=f"{n_det/len(live_df)*100:.1f}%",
                  delta_color="inverse")
        k3.metric("Clean",          f"{len(live_df)-n_det:,}")

        # ── Alert if botnet found ──────────────────────────────────────────
        if n_det > 0:
            st.error(f"ALERT: {n_det} botnet-like packets detected!")
            st.dataframe(
                live_df[live_df["predicted_label"] == 1][
                    ["SrcAddr", "DstAddr", "Proto", "TotBytes", "anomaly_score"]
                ].reset_index(drop=True),
                use_container_width=True,
            )
        else:
            st.success("No botnet traffic detected in this capture window.")

        # Full results table
        with st.expander("Show all captured packets"):
            st.dataframe(
                live_df[["SrcAddr", "DstAddr", "Proto", "TotBytes",
                          "sport_numeric", "dport_numeric",
                          "predicted_label", "anomaly_score"]].style.apply(
                    lambda row: ["background-color: #FFDCDC"
                                 if row["predicted_label"] == 1
                                 else "background-color: #D4EDDA"] * len(row), axis=1
                ),
                use_container_width=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Page: Export Report
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📄 Export Report":
    st.title("📄 Export PDF Report")

    data_files = list(DATA_DIR.glob("*.csv"))
    models, feature_names = load_models()

    if not data_files or models is None:
        st.warning("Run `python train.py` first to generate data and models.")
        st.stop()

    selected_file = st.selectbox("Dataset to report on:", data_files,
                                 format_func=lambda p: p.name)
    model_name    = st.selectbox("Model:", list(models.keys()))
    report_title  = st.text_input("Report Title", "Botnet Detection Analysis Report")
    author_name   = st.text_input("Author Name", "Student")

    if st.button("Generate PDF Report", type="primary"):
        from fpdf import FPDF
        import datetime

        df   = load_data(str(selected_file))
        X, y = get_X_y(df)
        model = models[model_name]

        for col in feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_names]

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X)[:, 1]
            y_pred  = (y_score >= 0.5).astype(int)
        else:
            y_pred  = np.where(model.predict(X) == -1, 1, 0)
            y_score = -model.decision_function(X)

        pr_auc  = average_precision_score(y, y_score)
        roc_auc = roc_auc_score(y, y_score)
        from sklearn.metrics import precision_score, recall_score, f1_score
        prec = precision_score(y, y_pred, zero_division=0)
        rec  = recall_score(y, y_pred, zero_division=0)
        f1   = f1_score(y, y_pred, zero_division=0)

        # Save temp charts
        tmp_cm  = MODELS_DIR / "_tmp_cm.png"
        tmp_roc = MODELS_DIR / "_tmp_roc.png"

        fig, ax = plt.subplots(figsize=(4, 3))
        cm = confusion_matrix(y, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=["Normal","Botnet"]).plot(ax=ax, colorbar=False)
        ax.set_title("Confusion Matrix")
        fig.tight_layout(); fig.savefig(tmp_cm, dpi=120); plt.close(fig)

        fpr, tpr, _ = roc_curve(y, y_score)
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.plot(fpr, tpr, color="#C44E52", lw=2, label=f"ROC-AUC={roc_auc:.3f}")
        ax2.plot([0,1],[0,1],"k--",lw=1); ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR")
        ax2.set_title("ROC Curve"); ax2.legend()
        fig2.tight_layout(); fig2.savefig(tmp_roc, dpi=120); plt.close(fig2)

        # ── Build PDF ──────────────────────────────────────────────────────
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 12, report_title, ln=True, align="C")

        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, f"Author: {author_name}   |   Date: {datetime.date.today()}   |   Model: {model_name}", ln=True, align="C")
        pdf.ln(4)
        pdf.set_draw_color(180, 180, 180)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(6)

        # Dataset summary
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "1. Dataset Summary", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"  File         : {selected_file.name}", ln=True)
        pdf.cell(0, 6, f"  Total Flows  : {len(df):,}", ln=True)
        pdf.cell(0, 6, f"  Botnet Flows : {int(y.sum()):,}  ({y.mean()*100:.2f}%)", ln=True)
        pdf.cell(0, 6, f"  Normal Flows : {int((y==0).sum()):,}", ln=True)
        pdf.ln(4)

        # Model performance
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "2. Model Performance", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for label, val in [("PR-AUC", pr_auc), ("ROC-AUC", roc_auc),
                            ("Precision", prec), ("Recall", rec), ("F1-Score", f1)]:
            pdf.cell(0, 6, f"  {label:<15}: {val:.4f}", ln=True)
        pdf.ln(4)

        # Charts
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "3. Visualisations", ln=True)
        pdf.image(str(tmp_cm),  x=12,  y=None, w=85)
        pdf.image(str(tmp_roc), x=110, y=pdf.get_y() - 68, w=95)
        pdf.ln(4)

        # Feature list
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "4. Features Used", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for feat in feature_names:
            pdf.cell(0, 5, f"  - {feat}", ln=True)
        pdf.ln(4)

        # Confusion matrix breakdown
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "5. Confusion Matrix Breakdown", ln=True)
        pdf.set_font("Helvetica", "", 10)
        tn, fp, fn, tp = cm.ravel()
        for label, val in [("True Negatives (correctly normal)", tn),
                            ("False Positives (normal flagged as botnet)", fp),
                            ("False Negatives (botnet missed)", fn),
                            ("True Positives (botnet caught)", tp)]:
            pdf.cell(0, 6, f"  {label:<45}: {val}", ln=True)

        # Conclusion
        pdf.ln(6)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "6. Conclusion", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6,
            f"The {model_name} model achieved a PR-AUC of {pr_auc:.4f} and ROC-AUC of "
            f"{roc_auc:.4f} on the {selected_file.name} dataset. "
            f"It correctly identified {tp} out of {tp+fn} botnet flows (Recall={rec:.2%}), "
            f"with {fp} false alarms out of {tn+fp} normal flows. "
            "This demonstrates effective botnet detection using machine learning on network flow data."
        )

        # Output
        pdf_bytes = bytes(pdf.output())
        st.success("PDF report generated!")
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name=f"botnet_report_{datetime.date.today()}.pdf",
            mime="application/pdf",
        )

        # Clean tmp files
        for f in [tmp_cm, tmp_roc]:
            if f.exists():
                f.unlink()


# ─────────────────────────────────────────────────────────────────────────────
# Page: About
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📖 About":

    st.title("📖 About this Project")
    st.markdown("""
## Botnet Detection System

This project detects **botnet network traffic** in real network flow data using
machine learning. It uses the **CTU-13 dataset** from Stratosphere Labs.

### What is a Botnet?
A botnet is a network of compromised machines (*bots*) controlled by an attacker
via a **Command & Control (C&C)** server. They are used for DDoS attacks, spam
campaigns, credential theft, and ransomware distribution.

### Detection Approach
| Layer | Technique |
|---|---|
| Feature Engineering | Velocity, byte ratios, port number, protocol |
| Supervised | Random Forest, Gradient Boosting, Decision Tree, Logistic Regression |
| Unsupervised | Isolation Forest (zero-day detection) |
| Evaluation | PR-AUC, ROC-AUC, Recall @ Precision threshold |

### Dataset: CTU-13
- 13 real botnet scenarios (Neris, Rbot, Murlo, NSIS, etc.)
- Labeled flows: **Botnet**, **Normal**, **Background**
- Download: https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/

### Tech Stack
- **Python**, **Scikit-learn**, **Pandas**, **NumPy**
- **Streamlit** (this dashboard), **Matplotlib**
""")
