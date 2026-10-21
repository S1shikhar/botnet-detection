"""
feature_engineering.py
-----------------------
Extracts and engineers features from CTU-13 binetflow CSV files.

CTU-13 binetflow columns:
  StartTime, Dur, Proto, SrcAddr, Sport, Dir, DstAddr, Dport,
  State, sTos, dTos, TotPkts, TotBytes, SrcBytes, Label
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# ── Columns we care about ────────────────────────────────────────────────────
FEATURE_COLS = [
    "Dur", "Proto", "Sport", "Dport", "State",
    "TotPkts", "TotBytes", "SrcBytes",
    # Engineered
    "bytes_per_pkt", "src_bytes_ratio", "dst_bytes",
    "sport_numeric", "dport_numeric",
]

PROTO_MAP = {"tcp": 0, "udp": 1, "icmp": 2, "arp": 3, "ipv6": 4}


def _parse_port(port_series: pd.Series) -> pd.Series:
    """Convert port values (may be hex or service names) to integers."""
    def convert(v):
        if pd.isna(v):
            return 0
        v = str(v).strip().lower()
        try:
            return int(v, 0)          # handles '0x1f90' → 8080
        except ValueError:
            known = {"http": 80, "https": 443, "dns": 53, "ssh": 22,
                     "ftp": 21, "smtp": 25, "pop3": 110, "imap": 143}
            return known.get(v, 0)
    return port_series.apply(convert)


def load_and_engineer(filepath: str) -> pd.DataFrame:
    """
    Load a CTU-13 binetflow file and return an engineered DataFrame
    with a binary column `is_botnet` (1 = botnet, 0 = normal/background).
    """
    df = pd.read_csv(filepath, low_memory=False)
    df.columns = df.columns.str.strip()

    # ── Label encoding ──────────────────────────────────────────────────────
    # CTU-13 labels include: 'flow=Background', 'flow=Normal', 'flow=To-Botnet-...'
    # We treat anything with 'Botnet' in the label as positive class.
    if "Label" not in df.columns:
        raise ValueError("No 'Label' column found — check the file format.")
    df["is_botnet"] = df["Label"].str.contains("Botnet", case=False, na=False).astype(int)

    # ── Numeric coercions ────────────────────────────────────────────────────
    df["Dur"]      = pd.to_numeric(df["Dur"],      errors="coerce").fillna(0)
    df["TotPkts"]  = pd.to_numeric(df["TotPkts"],  errors="coerce").fillna(0)
    df["TotBytes"] = pd.to_numeric(df["TotBytes"],  errors="coerce").fillna(0)
    df["SrcBytes"] = pd.to_numeric(df["SrcBytes"],  errors="coerce").fillna(0)

    # ── Protocol encoding ────────────────────────────────────────────────────
    df["Proto"] = (
        df["Proto"].str.lower().str.strip()
        .map(PROTO_MAP)
        .fillna(99)                    # 99 = unknown
        .astype(int)
    )

    # ── State encoding ───────────────────────────────────────────────────────
    le_state = LabelEncoder()
    df["State"] = le_state.fit_transform(
        df["State"].fillna("UNK").str.strip().str.upper()
    )

    # ── Port engineering ─────────────────────────────────────────────────────
    df["sport_numeric"] = _parse_port(df.get("Sport"))
    df["dport_numeric"] = _parse_port(df.get("Dport"))

    # ── Derived features ─────────────────────────────────────────────────────
    df["bytes_per_pkt"]    = df["TotBytes"] / (df["TotPkts"].replace(0, 1))
    df["src_bytes_ratio"]  = df["SrcBytes"] / (df["TotBytes"].replace(0, 1))
    df["dst_bytes"]        = df["TotBytes"] - df["SrcBytes"]

    # ── Clip / fill any remaining NaNs ───────────────────────────────────────
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def get_X_y(df: pd.DataFrame):
    """Return feature matrix X and label vector y."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()
    y = df["is_botnet"].copy()
    return X, y
