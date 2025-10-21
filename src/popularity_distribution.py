"""Utility script for computing interaction coverage by item popularity.

This module reads dataset files of the format used in ``src/data/*.txt``
where the first column is a user identifier followed by item/timestamp
pairs.  It reports what fraction of the total interactions are captured by
items in the top 10%, 20%, …, 100% popularity buckets.
"""

from __future__ import annotations

import argparse
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Tuple

logger = logging.getLogger(__name__)

def parse_line(line: str, line_number: int) -> Iterable[str]:
    """Extract item identifiers from a line of the dataset.

    Parameters
    ----------
    line:
        A raw line from the dataset file.
    line_number:
        The line number (1-indexed) for logging purposes.

    Yields
    ------
    str
        The item identifiers present in the line.
    """

    parts = line.strip().split()
    if not parts:
        return

    if len(parts) == 1:
        logger.debug("Line %d only contains a user identifier; skipping", line_number)
        return

    # 从第二列开始，提取每个项的 id（即冒号前的部分）
    interactions = parts[1:]
    for interaction in interactions:
        item_id = interaction.split(':')[0]  # 只提取 item_id，忽略时间戳
        yield item_id

# def parse_line(line: str, line_number: int) -> Iterable[str]:
#     """Extract item identifiers from a line of the dataset.
#
#     Parameters
#     ----------
#     line:
#         A raw line from the dataset file.
#     line_number:
#         The line number (1-indexed) for logging purposes.
#
#     Yields
#     ------
#     str
#         The item identifiers present in the line.
#     """
#
#     parts = line.strip().split()
#     if not parts:
#         return
#
#     if len(parts) == 1:
#         logger.debug("Line %d only contains a user identifier; skipping", line_number)
#         return
#
#     interactions = parts[1:]
#     if len(interactions) % 2 != 0:
#         logger.warning(
#             "Line %d has an odd number of tokens after the user id; "
#             "the last token will be ignored.",
#             line_number,
#         )
#         interactions = interactions[:-1]
#
#     for index in range(0, len(interactions), 2):
#         yield interactions[index]


def count_item_interactions(path: Path) -> Tuple[Counter, int]:
    """Count how many times each item appears in the dataset."""

    item_counts: Counter[str] = Counter()
    total = 0

    with path.open("r", encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            for item in parse_line(line, line_number):
                item_counts[item] += 1
                total += 1
                # break
    # print(item_counts)
    return item_counts, total


def compute_coverage(item_counts: Counter[str], total_interactions: int) -> Iterable[Tuple[int, float]]:
    """Compute coverage ratios for popularity percentiles."""

    if total_interactions == 0 or not item_counts:
        for percent in range(10, 101, 10):
            yield percent, 0.0
        return
    sorted_counts = sorted(item_counts.values(), reverse=True)
    prefix_sums = []
    cumulative = 0
    for count in sorted_counts:
        cumulative += count
        prefix_sums.append(cumulative)

    num_items = len(sorted_counts)
    for percent in [0.1,1,5,10,20, 50, 60,80,100]:
        top_k = max(1, math.ceil(percent / 100 * num_items))
        interactions_in_top = prefix_sums[top_k - 1]
        coverage = interactions_in_top / total_interactions
        yield percent, coverage


def format_results(results: Iterable[Tuple[int, float]]) -> str:
    """Format coverage ratios for display."""

    lines = ["Popularity Coverage"]
    lines.append("Percentile\tInteraction Share")
    for percent, coverage in results:
        lines.append(f"Top {percent}%\t{coverage:.4%}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the fraction of interactions covered by the most popular "
            "items at different popularity thresholds."
        )
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to the dataset file (e.g., src/data/Beauty.txt)",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity (default: WARNING).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    dataset_path: Path = args.dataset
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    item_counts, total_interactions = count_item_interactions(dataset_path)
    results = list(compute_coverage(item_counts, total_interactions))
    print(format_results(results))


if __name__ == "__main__":
    main()
