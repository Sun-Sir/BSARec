import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def load_interactions(path: Path):
    """Load interactions from a preprocessed txt file.

    The file format of each line should be:

        user item1:ts1 item2:ts2 ...

    where ``user`` is ignored and each ``item:timestamp`` pair is
    separated by whitespace.  Only item ids and their unix timestamps are
    used to construct the DataFrame.
    """

    items = []
    timestamps = []
    with path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) <= 1:
                continue
            # skip the user id at index 0
            for pair in parts[1:]:
                if ":" not in pair:
                    continue
                item_str, ts_str = pair.split(":", 1)
                try:
                    items.append(int(item_str))
                    timestamps.append(int(ts_str))
                except ValueError:
                    continue

    df = pd.DataFrame({"item": items, "timestamp": timestamps})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    return df


def discounted_cumsum(counts: np.ndarray, gamma: float) -> np.ndarray:
    """Compute discounted cumulative popularity for each item.

    Parameters
    ----------
    counts: 2D array where axis 0 is time (with a dummy first row) and
        axis 1 corresponds to item indices (dummy column at 0).
    gamma: discount factor applied to previous popularity.

    Returns
    -------
    np.ndarray
        Final discounted popularity per item including the dummy column.
    """

    pop = np.zeros(counts.shape[1], dtype=float)
    for row in range(1, counts.shape[0]):
        pop = pop * gamma + counts[row]
    return pop


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Convert raw popularity scores to percentile ranks (0-100)."""

    if len(values) == 0:
        return values
    ranks = pd.Series(values).rank(pct=True).to_numpy()
    return ranks * 100


def linear_encode(percentile: float, dim: int = 11) -> list:
    """Encode a percentile into a basis vector using linear interpolation."""

    step = 100 / (dim - 1)
    frac = percentile / step
    base = int(frac)
    vec = [0.0] * dim
    if base >= dim - 1:
        vec[-1] = 1.0
    else:
        weight_high = frac - base
        vec[base] = 1 - weight_high
        vec[base + 1] = weight_high
    return vec

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
    parser = argparse.ArgumentParser(
        description="Aggregate item popularity by month and week from preprocessed sequences."
    )
    parser.add_argument("input", type=Path, help="Path to txt file with user sequences")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name prefix for output files",
    )
    parser.add_argument(
        "--top_n", type=int, default=None, help="Keep only top N items by interaction count"
    )
    parser.add_argument(
        "--out_dir", type=Path, default=Path(__file__).parent, help="Directory to save results"
    )
    parser.add_argument(
        "--gamma_long", type=float, default=0.9, help="Discount factor for long-term popularity"
    )
    parser.add_argument(
        "--gamma_short", type=float, default=0.8, help="Discount factor for short-term popularity"
    )
    parser.add_argument(
        "--dim_long", type=int, default=11, help="Encoding dimension for long-term popularity"
    )
    parser.add_argument(
        "--dim_short", type=int, default=11, help="Encoding dimension for short-term popularity"
    )
    args = parser.parse_args()

    df = load_interactions(args.input)
    # build initial item map using all items
    all_items = sorted(df["item"].unique())
    item_map = {item: idx + 1 for idx, item in enumerate(all_items)}

    month_counts, month_map, item_map = aggregate_counts(df, "M", item_map, args.top_n)
    week_counts, week_map, item_map = aggregate_counts(df, "W", item_map, args.top_n)

    latest_week = week_counts.shape[0] - 1
    week_eval = np.zeros((2, week_counts.shape[1]), dtype=int)
    week_eval[1] = week_counts[latest_week]

    long_pop = discounted_cumsum(month_counts, args.gamma_long)[1:]
    short_pop = discounted_cumsum(week_counts, args.gamma_short)[1:]

    long_pct = percentile_rank(long_pop)
    short_pct = percentile_rank(short_pop)

    long_enc = [linear_encode(p, args.dim_long) for p in long_pct]
    short_enc = [linear_encode(p, args.dim_short) for p in short_pct]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        args.out_dir / f"{args.dataset}_month_pop.txt", month_counts, fmt="%d"
    )
    np.savetxt(
        args.out_dir / f"{args.dataset}_week_pop.txt", week_counts, fmt="%d"
    )
    np.savetxt(
        args.out_dir / f"{args.dataset}_week_eval_pop.txt", week_eval, fmt="%d"
    )

    pop_enc_path = args.out_dir / f"{args.dataset}_pop_linear.txt"
    with pop_enc_path.open("w") as f:
        for item_id, idx in sorted(item_map.items()):
            long_vec = [float(x) for x in long_enc[idx - 1]]
            short_vec = [float(x) for x in short_enc[idx - 1]]
            f.write(f"{item_id}\t{long_vec}\t{short_vec}\n")

if __name__ == "__main__":
    main()
