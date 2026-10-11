"""Build the results packet from every finished benchmark."""

import argparse
from pathlib import Path

from robustness_benchmark.results.brief import write_brief
from robustness_benchmark.results.figures import (
    plot_family_heatmaps,
    plot_metric_decomposition,
    plot_plausibility,
    plot_rank_diagrams,
    plot_tradeoff,
)
from robustness_benchmark.results.labels import MODEL_TYPE_LABELS, ordered
from robustness_benchmark.results.loading import discover_runs, load_results
from robustness_benchmark.results.rank_analysis import rank_test_table, run_rank_tests
from robustness_benchmark.results.tables import (
    cross_class_markdown,
    cross_class_summary,
    save_runtime_table,
    save_tables,
    seed_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/results_packet"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    results = load_results(discover_runs(args.artifacts))
    overall, family = results.overall, results.family

    save_tables(output, overall, family, results.seeds, seed_summary(results.seeds))
    save_runtime_table(output, results.runtime)
    cross_class = cross_class_summary(overall)
    cross_class.to_csv(output / "cross_class_summary.csv", index_label="method")
    (output / "cross_class_summary.md").write_text(cross_class_markdown(cross_class))
    rank_results = run_rank_tests(overall, family)
    rank_tests = rank_test_table(rank_results)
    rank_tests.to_csv(output / "rank_tests.csv", index=False)

    plot_rank_diagrams(output, rank_results)
    for model_type in ordered(overall["model_type"], tuple(MODEL_TYPE_LABELS)):
        plot_family_heatmaps(output, family, model_type)
        plot_tradeoff(output, overall, model_type)
        plot_plausibility(output, overall, model_type)
        plot_metric_decomposition(output, overall, model_type)
    write_brief(output, overall, family, cross_class, rank_tests)


if __name__ == "__main__":
    main()
