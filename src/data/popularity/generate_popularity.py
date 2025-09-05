import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

def load_interactions(path: Path, sep: str):
    """Load interactions with columns: user, item, rating, timestamp.
    The file is expected to be delimiter-separated without header."""
    df = pd.read_csv(path, sep=sep, engine="python", names=["user", "item", "rating", "timestamp"])
    df = df[["item", "timestamp"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    return df

def aggregate_counts(
    df: pd.DataFrame, freq: str, item_map: dict, top_n: Optional[int] = None
):
    """Aggregate interaction counts per item per period.

    Parameters
    ----------
    df: DataFrame with columns item and timestamp
    freq: Resampling frequency ('M' for month, 'W' for week)
    item_map: mapping from item id to column index (1-indexed)
    top_n: optional number of top items to retain (by total count)
    """
    if top_n is not None:
        top_items = df["item"].value_counts().nlargest(top_n).index
        df = df[df["item"].isin(top_items)]
        item_map = {item: idx + 1 for idx, item in enumerate(sorted(top_items))}
    # determine unique periods in chronological order
    periods = df["timestamp"].dt.to_period(freq).sort_values().unique()
    period_map = {p: idx + 1 for idx, p in enumerate(periods)}
    counts = np.zeros((len(periods) + 1, len(item_map) + 1), dtype=int)
    df["p_idx"] = df["timestamp"].dt.to_period(freq).map(period_map)
    df["i_idx"] = df["item"].map(item_map)
    for p, i in zip(df["p_idx"], df["i_idx"]):
        counts[p, i] += 1
    return counts, period_map, item_map

def main():
    parser = argparse.ArgumentParser(description="Aggregate item popularity by month and week.")
    parser.add_argument("input", type=Path, help="Path to raw interaction file")
    parser.add_argument("--sep", default="::", help="Field separator in the input file")
    parser.add_argument("--top_n", type=int, default=None, help="Keep only top N items by interaction count")
    parser.add_argument("--out_dir", type=Path, default=Path(__file__).parent, help="Directory to save results")
    args = parser.parse_args()

    df = load_interactions(args.input, args.sep)
    # build initial item map using all items
    all_items = sorted(df["item"].unique())
    item_map = {item: idx + 1 for idx, item in enumerate(all_items)}

    month_counts, month_map, item_map = aggregate_counts(df, "M", item_map, args.top_n)
    week_counts, week_map, item_map = aggregate_counts(df, "W", item_map, args.top_n)

    latest_week = week_counts.shape[0] - 1
    week_eval = np.zeros((2, week_counts.shape[1]), dtype=int)
    week_eval[1] = week_counts[latest_week]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.out_dir / "month_pop.txt", month_counts, fmt="%d")
    np.savetxt(args.out_dir / "week_pop.txt", week_counts, fmt="%d")
    np.savetxt(args.out_dir / "week_eval_pop.txt", week_eval, fmt="%d")

if __name__ == "__main__":
    main()
