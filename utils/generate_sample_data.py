"""
generate_sample_data.py
-----------------------
Generates a small synthetic binetflow-format CSV that mirrors CTU-13 structure.
Use this to run and test the project before downloading the real CTU-13 dataset.
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)


def _random_ip(private=True):
    if private:
        return f"192.168.{np.random.randint(0,5)}.{np.random.randint(1,255)}"
    return f"{np.random.randint(1,223)}.{np.random.randint(0,255)}.{np.random.randint(0,255)}.{np.random.randint(1,254)}"


def generate(n_normal=4000, n_botnet=400, out_path: str = "data/sample_ctu13.csv"):
    records = []

    # ── Normal traffic ────────────────────────────────────────────────────────
    for _ in range(n_normal):
        proto = np.random.choice(["tcp", "udp", "icmp"], p=[0.6, 0.35, 0.05])
        records.append({
            "StartTime": "2011/08/10 09:00:00.000000",
            "Dur":       round(np.random.exponential(5), 6),
            "Proto":     proto,
            "SrcAddr":   _random_ip(private=True),
            "Sport":     np.random.choice([1024 + i for i in range(40000)]),
            "Dir":       "->",
            "DstAddr":   _random_ip(private=False),
            "Dport":     np.random.choice([80, 443, 53, 22, 8080]),
            "State":     np.random.choice(["FIN", "CON", "INT", "RST"], p=[0.4,0.4,0.1,0.1]),
            "sTos":      0,
            "dTos":      0,
            "TotPkts":   np.random.randint(2, 50),
            "TotBytes":  np.random.randint(100, 100_000),
            "SrcBytes":  np.random.randint(50, 50_000),
            "Label":     np.random.choice(["flow=Normal", "flow=Background"], p=[0.5, 0.5]),
        })

    # ── Botnet traffic (C&C beaconing pattern) ────────────────────────────────
    # Characteristics: short duration, fixed dst port (IRC 6667 / HTTP), many tiny pkts
    c2_server = _random_ip(private=False)
    bot_host  = _random_ip(private=True)

    for _ in range(n_botnet):
        records.append({
            "StartTime": "2011/08/10 09:00:00.000000",
            "Dur":       round(np.random.uniform(0.001, 0.5), 6),   # very short
            "Proto":     "tcp",
            "SrcAddr":   bot_host,
            "Sport":     np.random.randint(49152, 65535),
            "Dir":       "->",
            "DstAddr":   c2_server,
            "Dport":     np.random.choice([6667, 80, 443]),          # IRC / HTTP C&C
            "State":     "CON",
            "sTos":      0,
            "dTos":      0,
            "TotPkts":   np.random.randint(2, 8),                   # very few packets
            "TotBytes":  np.random.randint(60, 600),                 # tiny payloads
            "SrcBytes":  np.random.randint(40, 300),
            "Label":     "flow=From-Botnet-V44-UDP-1",
        })

    df = pd.DataFrame(records)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[OK] Synthetic dataset saved -> {out_path}  ({len(df)} rows, {n_botnet} botnet)")
    return out_path


if __name__ == "__main__":
    generate()
