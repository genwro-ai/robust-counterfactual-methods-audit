"""Compare Statlog German Credit (UCI 144) with South German Credit (UCI 522).

South German Credit corrects the Statlog code table (Grömping 2019). This
script checks whether the stored values differ as well. Both files hold the
same 1,000 credits but in a different row order, so rows are aligned on the
three quantitative features (duration, amount, age) as in Grömping's report;
rows left over (typos in amount or age) are aligned on duration plus one of
the other two. Statlog codes are translated to South German codes first:
``A<attribute><code>`` becomes ``<code>``, people_liable and foreign_worker
are reversed (``3 - code``), and the label 1 = good / 2 = bad becomes 1 / 0.
"""

import argparse
from pathlib import Path

import pandas as pd

from robustness_benchmark.core.data import DATA_DIR

KEYS = (
    ("duration", "amount", "age"),
    ("duration", "amount"),
    ("duration", "age"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--south", type=Path, default=DATA_DIR / "german_credit.csv")
    parser.add_argument(
        "--statlog", type=Path, default=DATA_DIR / "german_statlog.data"
    )
    return parser.parse_args()


def statlog_in_south_codes(path: Path, columns: pd.Index) -> pd.DataFrame:
    statlog = pd.read_csv(path, sep=" ", header=None, names=columns)
    for position, column in enumerate(columns[:-1], start=1):
        if statlog[column].dtype == object:
            prefix = f"A{position}"
            assert statlog[column].str.startswith(prefix).all(), column
            statlog[column] = statlog[column].str.removeprefix(prefix).astype(int)
    for column in ("people_liable", "foreign_worker"):
        statlog[column] = 3 - statlog[column]
    statlog["credit_risk"] = statlog["credit_risk"].map({1: 1, 2: 0})
    return statlog


def align(south: pd.DataFrame, statlog: pd.DataFrame) -> pd.DataFrame:
    """Return (south row, statlog row) index pairs, one per credit."""

    pairs = []
    left, right = south, statlog
    for key in KEYS:
        if left.empty:
            break
        merged = left.reset_index().merge(
            right.reset_index(), on=list(key), suffixes=("_south", "_statlog")
        )
        if merged["index_south"].duplicated().any() or (
            merged["index_statlog"].duplicated().any()
        ):
            raise ValueError(f"Key {key} does not identify rows uniquely")
        pairs.append(merged[["index_south", "index_statlog"]])
        left = left.drop(index=merged["index_south"])
        right = right.drop(index=merged["index_statlog"])
    if not left.empty:
        raise ValueError(f"{len(left)} South German rows could not be aligned")
    return pd.concat(pairs, ignore_index=True)


def main() -> None:
    args = parse_args()
    south = pd.read_csv(args.south)
    statlog = statlog_in_south_codes(args.statlog, south.columns)
    print(f"rows: South German {len(south)}, Statlog {len(statlog)}")
    same_position = (south[list(KEYS[0])] == statlog[list(KEYS[0])]).all(axis=1)
    print(f"rows whose quantitative features agree by position: {same_position.sum()}")

    pairs = align(south, statlog)
    left = south.loc[pairs["index_south"]].reset_index(drop=True)
    right = statlog.loc[pairs["index_statlog"]].reset_index(drop=True)
    differs = left.ne(right)
    print(f"aligned rows: {len(pairs)}")
    print("rows with differing values, per column:")
    print(differs.sum()[differs.any()].to_string())
    print(f"rows with any difference: {differs.any(axis=1).sum()}")
    print(f"label differences: {differs['credit_risk'].sum()}")
    for column in differs.columns[differs.any()]:
        changed = differs[column]
        transitions = pd.crosstab(
            right.loc[changed, column].rename(f"{column} statlog"),
            left.loc[changed, column].rename("south"),
        )
        print(f"\n{column} (Statlog value -> South German value):")
        print(transitions.to_string())


if __name__ == "__main__":
    main()
