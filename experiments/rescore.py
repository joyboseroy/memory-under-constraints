"""Rescore a finished run under different prices for compute.

    python experiments/rescore.py results/E01_memory_budget/results.csv
    python experiments/rescore.py results/E01_memory_budget/results.csv --lams 0 0.001 0.01
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from muc.rescore import report  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("results_csv")
ap.add_argument("--lams", type=float, nargs="*", help="prices of compute, in nats of KL per kFLOP")
args = ap.parse_args()
print(report(pd.read_csv(args.results_csv), args.lams))
