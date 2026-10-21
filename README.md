# 🕵️ Botnet Detection System

A machine-learning-based botnet detection system for network flow data,
built for the **CTU-13 dataset** from Stratosphere Labs.

---

## Project Structure

```
botnet-detection/
├── app.py                     # Streamlit dashboard
├── train.py                   # Model training pipeline
├── requirements.txt
├── data/                      # Place CTU-13 binetflow CSVs here
├── models/                    # Saved models & charts (auto-generated)
├── utils/
│   ├── feature_engineering.py # Feature extraction from binetflow files
│   └── generate_sample_data.py# Synthetic data generator for quick testing
└── notebooks/
    └── exploration.ipynb      # EDA notebook (optional)
```

---

## Quick Start (5 minutes)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate sample data & train models
```bash
# Generate a small synthetic CTU-13-like dataset
python utils/generate_sample_data.py

# Train all models (uses data/sample_ctu13.csv by default)
python train.py
```

### 3. Launch the dashboard
```bash
streamlit run app.py
```

---

## Using the Real CTU-13 Dataset

1. Visit: https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/
2. Download **Scenario 10** (small, ~30 MB uncompressed):
   `capture20110818.pcap.netflow.labeled`
3. Place it in `data/` and run:

```bash
python train.py --data data/capture20110818.pcap.netflow.labeled
```

> **Tip:** Start with a single scenario. The full CTU-13 dataset is 1.9 GB compressed.

---

## Models Compared

| Model | Type | Key Advantage |
|---|---|---|
| **Random Forest** | Supervised | Best overall accuracy, interpretable |
| Gradient Boosting | Supervised | Handles class imbalance well |
| Decision Tree | Supervised | Fully explainable, fast |
| Logistic Regression | Supervised | Baseline, very fast |
| **Isolation Forest** | **Unsupervised** | Detects zero-day botnet patterns |

---

## Key Features Engineered

| Feature | Description |
|---|---|
| `Dur` | Duration of the network flow |
| `TotPkts` | Total packets in flow |
| `TotBytes` | Total bytes transferred |
| `bytes_per_pkt` | Average bytes per packet (low = C&C heartbeat) |
| `src_bytes_ratio` | Fraction of bytes from source |
| `velocity_score` | Inverse of time since last flow (high = rapid beaconing) |
| `sport_numeric` | Source port (numeric) |
| `dport_numeric` | Destination port (numeric) |

---

## Evaluation Metrics

> **Never use Accuracy** on this dataset — botnet traffic is < 1% of flows.
> Use PR-AUC (primary), ROC-AUC, and Recall @ fixed Precision.

---

## References

- CTU-13 Dataset: García et al., *An empirical comparison of botnet detection methods*, Computers & Security, 2014.
- Stratosphere Research Laboratory: https://www.stratosphereips.org/
