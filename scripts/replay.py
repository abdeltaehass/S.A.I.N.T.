"""
Replay NSL-KDD test samples through the live API.

Usage:
    python scripts/replay.py               # 50 samples, 0.1s delay
    python scripts/replay.py --n 200       # 200 samples
    python scripts/replay.py --delay 0.05  # faster
    python scripts/replay.py --n 0         # all samples
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import FEATURE_NAMES, FLASK_PORT

API_URL = f"http://localhost:{FLASK_PORT}/predict"
DATA_FILE = Path(__file__).resolve().parents[1] / "data/raw/KDDTest+.txt"
COLUMNS = FEATURE_NAMES + ["attack_type", "difficulty"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",     type=int,   default=50,  help="samples to send (0 = all)")
    parser.add_argument("--delay", type=float, default=0.1, help="seconds between requests")
    args = parser.parse_args()

    df = pd.read_csv(DATA_FILE, header=None, names=COLUMNS)
    if args.n > 0:
        df = df.sample(n=min(args.n, len(df)), random_state=42)

    print(f"Replaying {len(df)} samples → {API_URL}  (delay={args.delay}s)\n")

    for i, (_, row) in enumerate(df.iterrows(), 1):
        payload = {col: row[col] for col in FEATURE_NAMES}
        true_label = row["attack_type"].strip()

        try:
            resp = requests.post(API_URL, json=payload, timeout=5)
            d = resp.json()
            action_color = {"allow": "\033[92m", "flag": "\033[93m", "block": "\033[91m"}
            color = action_color.get(d["action"], "")
            reset = "\033[0m"
            print(
                f"[{i:>4}] true={true_label:<20} pred={d['predicted_class']:<8} "
                f"conf={d['confidence']:.3f}  {color}{d['action'].upper()}{reset}"
            )
        except requests.RequestException as e:
            print(f"[{i:>4}] ERROR: {e}")

        time.sleep(args.delay)

    print("\nDone. Check the dashboard at http://localhost:8050")


if __name__ == "__main__":
    main()
