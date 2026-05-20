from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from scipy.stats import friedmanchisquare, rankdata, wilcoxon
except ModuleNotFoundError:
    friedmanchisquare = None
    rankdata = None
    wilcoxon = None

try:
    import scikit_posthocs as sp
except ModuleNotFoundError:
    sp = None


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis"
PLOTS_DIR = ANALYSIS_DIR / "plots"
TABLES_DIR = ANALYSIS_DIR / "tables"

DEFAULT_RESULT_FILES = [
    ROOT / "results" / "opro" / "opro_results.json",
    ROOT / "results" / "spso2011" / "spso2011_results.json",
    ROOT / "results" / "lshade" / "lshade_results.json",
    ROOT / "results" / "randomsearch" / "randomsearch_results.json",
]

PATH_FALLBACKS = {
    ROOT / "results" / "randomsearch" / "randomsearch_results.json": ROOT
    / "results"
    / "random_search"
    / "randomsearch_results.json",
}

SUMMARY_CSV = TABLES_DIR / "summary_table.csv"
SUMMARY_MD = TABLES_DIR / "summary_table.md"
NEMENYI_CSV = TABLES_DIR / "nemenyi_matrix.csv"
WILCOXON_CSV = TABLES_DIR / "wilcoxon_results.csv"
RANKS_CSV = TABLES_DIR / "method_ranks.csv"


def _resolve_result_path(path: Path) -> Path:
    if path.exists():
        return path
    fallback = PATH_FALLBACKS.get(path)
    if fallback is not None and fallback.exists():
        return fallback
    raise FileNotFoundError(f"Missing aggregate JSON: {path}")


def _load_json(path: Path) -> dict[str, Any]:
    resolved = _resolve_result_path(path)
    with resolved.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_methods(paths: list[Path] | None = None) -> dict[str, dict[str, Any]]:
    methods_data: dict[str, dict[str, Any]] = {}
    for path in paths or DEFAULT_RESULT_FILES:
        payload = _load_json(path)
        method = str(payload["method"])
        if method in methods_data:
            raise ValueError(f"Duplicate method in inputs: {method}")
        methods_data[method] = payload
    return methods_data


def _final_j_arrays(methods_data: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
    arrays = {
        method: np.asarray(payload["final_J"], dtype=float)
        for method, payload in methods_data.items()
    }
    lengths = {method: values.shape[0] for method, values in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"final_J lengths differ across methods: {lengths}")
    return arrays


def compute_descriptive_stats(
    methods_data: dict[str, dict[str, Any]],
) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    for method, payload in methods_data.items():
        values = np.asarray(payload["final_J"], dtype=float)
        q25, q75 = np.percentile(values, [25, 75])
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        rows.append(
            {
                "method": method,
                "n_runs": int(values.size),
                "mean": mean,
                "median": float(np.median(values)),
                "std": std,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "IQR": float(q75 - q25),
                "CV": float(std / abs(mean)) if mean != 0.0 else float("nan"),
            }
        )
    return rows


def write_summary_tables(rows: list[dict[str, float | str | int]]) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method", "n_runs", "mean", "median", "std", "min", "max", "IQR", "CV"]
    _write_csv(SUMMARY_CSV, rows, fieldnames)

    with SUMMARY_MD.open("w", encoding="utf-8") as file:
        file.write("| " + " | ".join(fieldnames) + " |\n")
        file.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for row in rows:
            values = [_format_table_value(row[name]) for name in fieldnames]
            file.write("| " + " | ".join(values) + " |\n")


def friedman_test(methods_data: dict[str, dict[str, Any]]) -> tuple[float, float]:
    if friedmanchisquare is None:
        raise ModuleNotFoundError("scipy is required for Friedman test")

    arrays = _final_j_arrays(methods_data)
    result = friedmanchisquare(*arrays.values())
    return float(result.statistic), float(result.pvalue)


def nemenyi_posthoc(methods_data: dict[str, dict[str, Any]]) -> np.ndarray:
    if sp is None:
        raise ModuleNotFoundError("scikit-posthocs is required for Nemenyi post-hoc")

    arrays = _final_j_arrays(methods_data)
    matrix = np.column_stack([arrays[method] for method in methods_data])
    result = sp.posthoc_nemenyi_friedman(matrix)
    p_values = np.asarray(result, dtype=float)
    _write_matrix_csv(NEMENYI_CSV, list(methods_data.keys()), p_values)
    return p_values


def compute_ranks(methods_data: dict[str, dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    if rankdata is None:
        raise ModuleNotFoundError("scipy is required for rank computation")

    arrays = _final_j_arrays(methods_data)
    matrix = np.column_stack([arrays[method] for method in methods_data])
    ranks = np.apply_along_axis(rankdata, 1, matrix, method="average")
    average_ranks = np.mean(ranks, axis=0)
    return ranks, average_ranks


def write_method_ranks(
    methods_data: dict[str, dict[str, Any]],
    ranks: np.ndarray,
    average_ranks: np.ndarray,
) -> None:
    methods = list(methods_data.keys())
    rows: list[dict[str, float | int | str]] = []
    for run_idx, run_ranks in enumerate(ranks):
        row: dict[str, float | int | str] = {"run_idx": run_idx}
        row.update({method: float(rank) for method, rank in zip(methods, run_ranks)})
        rows.append(row)

    avg_row: dict[str, float | int | str] = {"run_idx": "average"}
    avg_row.update(
        {method: float(rank) for method, rank in zip(methods, average_ranks)}
    )
    rows.append(avg_row)

    _write_csv(RANKS_CSV, rows, ["run_idx", *methods])


def wilcoxon_tests(methods_data: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if wilcoxon is None:
        raise ModuleNotFoundError("scipy is required for Wilcoxon tests")

    arrays = _final_j_arrays(methods_data)
    rows: list[dict[str, Any]] = []
    for method_a, method_b in itertools.combinations(methods_data.keys(), 2):
        result = wilcoxon(arrays[method_a], arrays[method_b], zero_method="wilcox")
        rows.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "statistic": float(result.statistic),
                "p_value": float(result.pvalue),
            }
        )

    adjusted = _holm_adjust([float(row["p_value"]) for row in rows])
    for row, adjusted_p in zip(rows, adjusted):
        row["p_holm"] = adjusted_p
        row["reject_holm_0.05"] = bool(adjusted_p <= 0.05)

    _write_csv(
        WILCOXON_CSV,
        rows,
        ["method_a", "method_b", "statistic", "p_value", "p_holm", "reject_holm_0.05"],
    )
    return rows


def plot_boxplot(methods_data: dict[str, dict[str, Any]]) -> None:
    methods, values = _plot_values(methods_data)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.boxplot(values, labels=methods, showmeans=True)
    ax.set_ylabel("Final objective J")
    ax.set_title("Final Objective Distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "boxplot_final_J.png", dpi=250)
    plt.close(fig)


def plot_violin(methods_data: dict[str, dict[str, Any]]) -> None:
    methods, values = _plot_values(methods_data)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    parts = ax.violinplot(values, showmeans=True, showmedians=True)
    for body in parts["bodies"]:
        body.set_alpha(0.55)
    ax.set_xticks(np.arange(1, len(methods) + 1))
    ax.set_xticklabels(methods)
    ax.set_ylabel("Final objective J")
    ax.set_title("Final Objective Density")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "violin_final_J.png", dpi=250)
    plt.close(fig)


def plot_convergence(methods_data: dict[str, dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for method, payload in methods_data.items():
        histories = _padded_histories(payload["histories"])
        median = np.median(histories, axis=0)
        q25 = np.percentile(histories, 25, axis=0)
        q75 = np.percentile(histories, 75, axis=0)
        x = np.arange(1, median.size + 1)
        ax.plot(x, median, label=method, linewidth=1.8)
        ax.fill_between(x, q25, q75, alpha=0.18)

    ax.set_xlabel("Objective evaluations")
    ax.set_ylabel("Best objective J")
    ax.set_title("Median Convergence with IQR Band")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "convergence.png", dpi=250)
    plt.close(fig)


def plot_cd_diagram(
    methods_data: dict[str, dict[str, Any]],
    average_ranks: np.ndarray,
    nemenyi_p_values: np.ndarray,
    alpha: float = 0.05,
) -> None:
    methods = list(methods_data.keys())
    order = np.argsort(average_ranks)
    sorted_methods = [methods[idx] for idx in order]
    sorted_ranks = average_ranks[order]

    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    y_axis = 0.7
    ax.hlines(y_axis, sorted_ranks.min() - 0.25, sorted_ranks.max() + 0.25, color="black")
    for method, rank in zip(sorted_methods, sorted_ranks):
        ax.vlines(rank, y_axis - 0.04, y_axis + 0.04, color="black")
        ax.text(rank, y_axis - 0.12, f"{rank:.2f}", ha="center", va="top", fontsize=9)
        ax.text(rank, y_axis + 0.12, method, ha="center", va="bottom", fontsize=10)

    groups = _non_significant_groups(order, nemenyi_p_values, alpha)
    for group_idx, group in enumerate(groups):
        if len(group) < 2:
            continue
        ranks = [average_ranks[idx] for idx in group]
        y = 0.42 - group_idx * 0.08
        ax.hlines(y, min(ranks), max(ranks), linewidth=2.0, color="black")

    ax.set_title("Critical Difference Diagram (Nemenyi, alpha=0.05)")
    ax.set_xlabel("Average rank (lower is better)")
    ax.set_yticks([])
    ax.set_xlim(sorted_ranks.min() - 0.35, sorted_ranks.max() + 0.35)
    ax.set_ylim(0.1, 1.05)
    ax.spines[["left", "right", "top"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "cd_diagram.png", dpi=250)
    plt.close(fig)


def save_plots(
    methods_data: dict[str, dict[str, Any]],
    average_ranks: np.ndarray,
    nemenyi_p_values: np.ndarray,
) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_boxplot(methods_data)
    plot_violin(methods_data)
    plot_convergence(methods_data)
    plot_cd_diagram(methods_data, average_ranks, nemenyi_p_values)


def main() -> int:
    try:
        methods_data = load_methods()
        summary_rows = compute_descriptive_stats(methods_data)
        write_summary_tables(summary_rows)
        friedman_statistic, friedman_p_value = friedman_test(methods_data)
        nemenyi_p_values = nemenyi_posthoc(methods_data)
        ranks, average_ranks = compute_ranks(methods_data)
        write_method_ranks(methods_data, ranks, average_ranks)
        wilcoxon_tests(methods_data)
        save_plots(methods_data, average_ranks, nemenyi_p_values)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    methods = list(methods_data.keys())
    best_rank_idx = int(np.argmin(average_ranks))
    print("Loaded methods:")
    for method in methods:
        print(f"- {method}")
    print()
    print(f"Friedman statistic = {friedman_statistic:.6g}")
    print(f"Friedman p-value = {friedman_p_value:.6g}")
    print(f"Best mean rank = {methods[best_rank_idx]} ({average_ranks[best_rank_idx]:.6g})")
    print(f"Plots saved to {PLOTS_DIR.relative_to(ROOT)}/")
    return 0


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_matrix_csv(path: Path, methods: list[str], matrix: np.ndarray) -> None:
    rows = []
    for method, values in zip(methods, matrix):
        row: dict[str, float | str] = {"method": method}
        row.update({name: float(value) for name, value in zip(methods, values)})
        rows.append(row)
    _write_csv(path, rows, ["method", *methods])


def _format_table_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _holm_adjust(p_values: list[float]) -> list[float]:
    n_values = len(p_values)
    adjusted = [0.0] * n_values
    running_max = 0.0
    for rank_idx, original_idx in enumerate(np.argsort(p_values)):
        value = min(1.0, p_values[original_idx] * (n_values - rank_idx))
        running_max = max(running_max, value)
        adjusted[original_idx] = running_max
    return adjusted


def _plot_values(
    methods_data: dict[str, dict[str, Any]],
) -> tuple[list[str], list[np.ndarray]]:
    methods = list(methods_data.keys())
    values = [
        np.asarray(methods_data[method]["final_J"], dtype=float)
        for method in methods
    ]
    return methods, values


def _padded_histories(histories: list[list[float]]) -> np.ndarray:
    max_len = max(len(history) for history in histories)
    padded = []
    for history in histories:
        if not history:
            raise ValueError("Empty history found")
        values = [float(value) for value in history]
        values.extend([values[-1]] * (max_len - len(values)))
        padded.append(values)
    return np.asarray(padded, dtype=float)


def _non_significant_groups(
    sorted_indices: np.ndarray,
    p_values: np.ndarray,
    alpha: float,
) -> list[list[int]]:
    groups: list[list[int]] = []
    n_methods = len(sorted_indices)
    for start in range(n_methods):
        best_group: list[int] = []
        for end in range(start + 1, n_methods + 1):
            candidate = list(sorted_indices[start:end])
            if all(
                p_values[i, j] >= alpha
                for i, j in itertools.combinations(candidate, 2)
            ):
                best_group = candidate
        if len(best_group) >= 2 and best_group not in groups:
            groups.append(best_group)
    return groups


if __name__ == "__main__":
    raise SystemExit(main())
