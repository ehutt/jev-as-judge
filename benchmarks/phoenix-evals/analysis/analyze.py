# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "arize-phoenix-client==3.5.0",
#   "httpx>=0.27",
#   "kaleido>=1.0",
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "scipy>=1.13",
# ]
# ///

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
from collections.abc import Iterable, Mapping
from itertools import combinations
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from phoenix.client import Client
from scipy.stats import binomtest, bootstrap, chi2, rankdata, spearmanr

JUDGE_ORDER = [
    "jev-slot-in",
    "jev-native",
    "openai-cheap",
    "openai-frontier",
    "anthropic-cheap",
    "anthropic-frontier",
    "gemini",
]

EXCLUSIONS_FILE = Path(__file__).with_name("exclusions.json")


def load_exclusions() -> tuple[
    dict[tuple[str, str], str],
    dict[tuple[str, str], str],
]:
    records = json.loads(EXCLUSIONS_FILE.read_text())
    by_input_text = {
        (record["evaluator"], record["input_text"]): record["reason"]
        for record in records
        if "input_text" in record
    }
    by_example_id = {
        (record["evaluator"], record["example_id"]): record["reason"]
        for record in records
        if "example_id" in record
    }
    return by_input_text, by_example_id


EXCLUSION_REASON_BY_INPUT_TEXT, EXCLUSION_REASON_BY_EXAMPLE_ID = load_exclusions()


def get_exclusion_reason(
    *, evaluator: str, example_id: str, input_text: object
) -> str | None:
    if reason := EXCLUSION_REASON_BY_EXAMPLE_ID.get((evaluator, example_id)):
        return reason
    if isinstance(input_text, str):
        return EXCLUSION_REASON_BY_INPUT_TEXT.get((evaluator, input_text))
    return None


def apply_exclusions(runs: pd.DataFrame) -> pd.DataFrame:
    runs = runs.copy()
    if "exclusion_reason" not in runs:
        runs["exclusion_reason"] = None
    configured_reasons = runs.apply(
        lambda row: get_exclusion_reason(
            evaluator=str(row["evaluator"]),
            example_id=str(row["example_id"]),
            input_text=row.get("input_text"),
        ),
        axis=1,
    )
    runs["exclusion_reason"] = configured_reasons.combine_first(runs["exclusion_reason"])
    return runs

EXPERIMENT_RUNS_QUERY = """
query JudgeSweepExperimentRuns($experimentId: ID!, $after: String) {
  experiment: node(id: $experimentId) {
    __typename
    ... on Experiment {
      runs(first: 100, after: $after) {
        edges {
          node {
            id
            repetitionNumber
            example { id }
            costSummary {
              prompt { tokens cost }
              completion { tokens cost }
              total { tokens cost }
            }
          }
        }
        pageInfo { endCursor hasNextPage }
      }
    }
  }
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build CSV, HTML, and PNG reports for one judge sweep."
    )
    parser.add_argument("sweep_run_id", help="The SWEEP_RUN_ID stored in experiment metadata")
    parser.add_argument(
        "--phoenix-endpoint",
        default=None,
        help="Phoenix base URL. Defaults to PHOENIX_ENDPOINT or localhost:6006.",
    )
    parser.add_argument(
        "--results-directory",
        type=Path,
        default=Path(__file__).parent / "results",
    )
    parser.add_argument(
        "--runs-csv",
        type=Path,
        default=None,
        help="Read a previous runs.csv instead of downloading runs from Phoenix.",
    )
    return parser.parse_args()


def get_base_url(explicit: str | None) -> str:
    configured = (
        explicit
        or os.getenv("PHOENIX_ENDPOINT")
        or os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
        or "http://localhost:6006"
    )
    return re.sub(r"/v1/(?:traces|logs|metrics)/?$", "", configured).rstrip("/")


def get_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    if configured := os.getenv("PHOENIX_CLIENT_HEADERS"):
        parsed = json.loads(configured)
        if not isinstance(parsed, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in parsed.items()
        ):
            raise ValueError("PHOENIX_CLIENT_HEADERS must be a JSON object of strings")
        headers.update(parsed)
    if api_key := os.getenv("PHOENIX_API_KEY"):
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def post_graphql(
    *, client: httpx.Client, query: str, variables: Mapping[str, Any]
) -> Mapping[str, Any]:
    response = client.post("/graphql", json={"query": query, "variables": variables})
    response.raise_for_status()
    body = response.json()
    if errors := body.get("errors"):
        messages = "\n".join(str(error.get("message", error)) for error in errors)
        raise RuntimeError(messages)
    data = body.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("Phoenix GraphQL response did not contain data")
    return data


def get_run_costs(
    *, client: httpx.Client, experiment_id: str
) -> dict[tuple[str, int], Mapping[str, Any]]:
    costs: dict[tuple[str, int], Mapping[str, Any]] = {}
    cursor: str | None = None
    while True:
        data = post_graphql(
            client=client,
            query=EXPERIMENT_RUNS_QUERY,
            variables={"experimentId": experiment_id, "after": cursor},
        )
        experiment = data.get("experiment")
        if not isinstance(experiment, dict):
            raise RuntimeError(f"Experiment {experiment_id} was not found")
        runs = experiment.get("runs")
        if not isinstance(runs, dict):
            raise RuntimeError(f"Experiment {experiment_id} has no runs connection")
        for edge in runs.get("edges", []):
            node = edge.get("node", {})
            example = node.get("example", {})
            key = (str(example.get("id")), int(node.get("repetitionNumber", 0)))
            costs[key] = node.get("costSummary", {})
        page_info = runs.get("pageInfo", {})
        if not page_info.get("hasNextPage"):
            break
        cursor = page_info.get("endCursor")
    return costs


def get_json(*, client: httpx.Client, path: str) -> list[dict[str, Any]]:
    response = client.get(path)
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, list):
        raise RuntimeError(f"Expected a JSON array from {path}")
    return value


def find_annotation(record: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    annotations = record.get("annotations", [])
    if not isinstance(annotations, list):
        return None
    return next(
        (
            annotation
            for annotation in annotations
            if isinstance(annotation, dict) and annotation.get("name") == name
        ),
        None,
    )


def classify_failure(*, error: Any, label: Any) -> str | None:
    if error:
        message = str(error).lower()
        if "token" in message and any(term in message for term in ("limit", "long", "maximum")):
            return "state_too_large"
        if "invalid label" in message or "unknown option" in message:
            return "invalid_label"
        if any(term in message for term in ("429", "rate limit", "timeout", "connection")):
            return "transport"
        return "evaluation_error"
    if not isinstance(label, str):
        return "missing_label"
    return None


def as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, dict) else {}


def get_price_snapshot(experiment: Mapping[str, Any]) -> tuple[float, float]:
    """The per-million-token rates the suite stamped on the experiment at run time."""
    pricing = as_mapping(as_mapping(experiment.get("metadata")).get("pricing"))
    input_rate = pricing.get("inputPerMillionTokensUsd")
    output_rate = pricing.get("outputPerMillionTokensUsd")
    if not isinstance(input_rate, (int, float)) or not isinstance(output_rate, (int, float)):
        raise RuntimeError(
            f"Experiment {experiment.get('id')} has no pricing snapshot in its metadata"
        )
    return float(input_rate), float(output_rate)


def build_run_row(
    *,
    evaluator: str,
    experiment: Mapping[str, Any],
    record: Mapping[str, Any],
    run_costs: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, Any] | None:
    reference = as_mapping(record.get("reference_output"))
    expected = reference.get("label")
    if not isinstance(expected, str):
        return None

    output = as_mapping(record.get("output"))
    record_input = as_mapping(record.get("input"))
    input_text = record_input.get("input")
    metadata = as_mapping(output.get("metadata"))
    probabilities = as_mapping(metadata.get("probabilities"))
    label = output.get("label")
    repetition = int(record.get("repetition_number", 0))
    example_id = str(record.get("example_id"))
    accuracy = find_annotation(record, "accuracy")
    error = record.get("error")
    failure_class = classify_failure(error=error, label=label)
    correct = bool(accuracy.get("score")) if accuracy else label == expected
    if failure_class is not None:
        correct = False

    experiment_metadata = as_mapping(experiment.get("metadata"))
    judge = str(experiment_metadata.get("judge"))
    input_rate, output_rate = get_price_snapshot(experiment)
    cost_summary = run_costs.get((example_id, repetition), {})
    prompt_cost_summary = as_mapping(cost_summary.get("prompt"))
    completion_cost_summary = as_mapping(cost_summary.get("completion"))
    prompt_tokens = record.get("prompt_token_count")
    if prompt_tokens is None:
        prompt_tokens = prompt_cost_summary.get("tokens")
    completion_tokens = record.get("completion_token_count")
    if completion_tokens is None:
        completion_tokens = completion_cost_summary.get("tokens")
    prompt_tokens_number = float(prompt_tokens or 0)
    completion_tokens_number = float(completion_tokens or 0)
    cost_check = (
        prompt_tokens_number * input_rate + completion_tokens_number * output_rate
    ) / 1_000_000

    total_cost = as_mapping(cost_summary.get("total")).get("cost")
    confidence = metadata.get("confidence")
    p_label = probabilities.get(label) if isinstance(label, str) else None
    return {
        "evaluator": evaluator,
        "judge": judge,
        "provider": experiment_metadata.get("provider"),
        "variant": experiment_metadata.get("variant"),
        "model_id": experiment_metadata.get("modelId"),
        "resolved_model": metadata.get("resolvedModel"),
        "experiment_id": experiment.get("id"),
        "example_id": example_id,
        "repetition": repetition,
        "label": label,
        "expected": expected,
        "correct": correct,
        "latency_ms": record.get("latency_ms"),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens_number + completion_tokens_number,
        "cost_usd": total_cost,
        "cost_check_usd": cost_check,
        "cost_delta_usd": None if total_cost is None else float(total_cost) - cost_check,
        "confidence": confidence,
        "p_label": p_label,
        "failure_class": failure_class,
        "error": error,
        "input_text": input_text,
        "exclusion_reason": get_exclusion_reason(
            evaluator=evaluator,
            example_id=example_id,
            input_text=input_text,
        ),
    }


def load_runs(*, sweep_run_id: str, base_url: str) -> pd.DataFrame:
    phoenix = Client(base_url=base_url, api_key=os.getenv("PHOENIX_API_KEY"))
    rows: list[dict[str, Any]] = []
    with httpx.Client(base_url=base_url, headers=get_headers(), timeout=120) as http:
        for dataset in phoenix.datasets.list():
            dataset_name = str(dataset.get("name", ""))
            if not dataset_name.startswith("judge-sweep/"):
                continue
            evaluator = dataset_name.removeprefix("judge-sweep/")
            for experiment in phoenix.experiments.list(dataset_id=dataset["id"]):
                metadata = as_mapping(experiment.get("metadata"))
                if metadata.get("sweepRunId") != sweep_run_id:
                    continue
                run_costs = get_run_costs(client=http, experiment_id=experiment["id"])
                records = get_json(client=http, path=f"/v1/experiments/{experiment['id']}/json")
                for record in records:
                    row = build_run_row(
                        evaluator=evaluator,
                        experiment=experiment,
                        record=record,
                        run_costs=run_costs,
                    )
                    if row is not None:
                        rows.append(row)
    if not rows:
        raise RuntimeError(f"No experiments found for sweep run id {sweep_run_id!r}")
    return pd.DataFrame(rows)


def select_best_experiments(runs: pd.DataFrame) -> pd.DataFrame:
    candidates = (
        runs.assign(has_error=runs["error"].notna())
        .groupby(["judge", "evaluator", "experiment_id"], as_index=False)
        .agg(rows=("example_id", "size"), errors=("has_error", "sum"))
        .sort_values(
            ["judge", "evaluator", "rows", "errors", "experiment_id"],
            ascending=[True, True, False, True, False],
        )
    )
    selected_ids = set(
        candidates.drop_duplicates(["judge", "evaluator"], keep="first")["experiment_id"]
    )
    return runs[runs["experiment_id"].isin(selected_ids)].copy()


def summarize_exclusions(excluded_runs: pd.DataFrame) -> pd.DataFrame:
    return (
        excluded_runs.groupby(
            ["evaluator", "example_id", "input_text", "expected", "exclusion_reason"],
            as_index=False,
            dropna=False,
        )
        .agg(judges=("judge", "nunique"), runs=("judge", "size"))
        .sort_values(["evaluator", "example_id"])
    )


def label_entropy_bits(labels: pd.Series) -> float:
    probabilities = labels.fillna("<failure>").astype(str).value_counts(normalize=True)
    return float(-(probabilities * np.log2(probabilities)).sum())


def bootstrap_accuracy(group: pd.DataFrame) -> tuple[float, float]:
    """95% interval from resampling examples; each example contributes its mean over repetitions."""
    accuracy_by_example = group.groupby("example_id")["correct"].mean().to_numpy(dtype=float)
    if len(accuracy_by_example) < 2 or np.all(accuracy_by_example == accuracy_by_example[0]):
        value = float(accuracy_by_example.mean())
        return value, value
    interval = bootstrap(
        (accuracy_by_example,),
        np.mean,
        confidence_level=0.95,
        n_resamples=5_000,
        method="percentile",
        rng=np.random.default_rng(20260921),
    ).confidence_interval
    return float(interval.low), float(interval.high)


def precision_recall_f1(
    *, truth: pd.Series, predicted: pd.Series, label: str
) -> tuple[float, float, float]:
    true_positive = int(((truth == label) & (predicted == label)).sum())
    false_positive = int(((truth != label) & (predicted == label)).sum())
    false_negative = int(((truth == label) & (predicted != label)).sum())
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def classification_metrics(group: pd.DataFrame) -> tuple[float, float, float, float]:
    truth = group["expected"].astype(str)
    predicted = group["label"].fillna("<failure>").astype(str)
    labels = sorted(truth.unique())
    if len(labels) < 2:
        return math.nan, math.nan, math.nan, math.nan

    scores = [
        precision_recall_f1(truth=truth, predicted=predicted, label=label) for label in labels
    ]

    observed = float((truth == predicted).mean())
    all_labels = sorted(set(truth) | set(predicted))
    expected_agreement = sum(
        float((truth == label).mean()) * float((predicted == label).mean()) for label in all_labels
    )
    kappa = (
        (observed - expected_agreement) / (1 - expected_agreement)
        if expected_agreement < 1
        else 1.0
    )
    return (
        float(np.mean([precision for precision, _, _ in scores])),
        float(np.mean([recall for _, recall, _ in scores])),
        float(np.mean([f1 for _, _, f1 in scores])),
        kappa,
    )


def summarize_class_balance(runs: pd.DataFrame) -> pd.DataFrame:
    examples = runs[["evaluator", "example_id", "expected"]].drop_duplicates()
    conflicts = examples.duplicated(["evaluator", "example_id"], keep=False)
    if conflicts.any():
        raise ValueError("An example has multiple expected labels within one evaluator")

    balance = (
        examples.groupby(["evaluator", "expected"], as_index=False)
        .size()
        .rename(columns={"size": "base_examples", "expected": "label"})
    )
    balance["expected_share"] = balance["base_examples"] / balance.groupby("evaluator")[
        "base_examples"
    ].transform("sum")
    balance["expected_class_count"] = balance.groupby("evaluator")["label"].transform("size")
    balance["classification_metrics_available"] = balance["expected_class_count"] > 1
    return balance


def summarize_class_metrics(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (evaluator, judge), group in runs.groupby(["evaluator", "judge"], sort=True):
        truth = group["expected"].astype(str)
        predicted = group["label"].fillna("<failure>").astype(str)
        labels = sorted(truth.unique())
        metrics_available = len(labels) > 1
        for label in labels:
            precision, recall, f1 = (
                precision_recall_f1(truth=truth, predicted=predicted, label=label)
                if metrics_available
                else (math.nan, math.nan, math.nan)
            )
            expected_rows = truth == label
            rows.append(
                {
                    "evaluator": evaluator,
                    "judge": judge,
                    "label": label,
                    "base_examples": int(group.loc[expected_rows, "example_id"].nunique()),
                    "runs": int(expected_rows.sum()),
                    "expected_share": float(expected_rows.mean()),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "metrics_available": metrics_available,
                }
            )
    return pd.DataFrame(rows)


def consistency_metrics(group: pd.DataFrame) -> tuple[float, float]:
    agreements: list[float] = []
    flips: list[bool] = []
    for _, example_runs in group.groupby("example_id"):
        labels = example_runs["label"].fillna("<failure>").astype(str)
        majority_label = labels.value_counts().index[0]
        agreements.append(float((labels == majority_label).mean()))
        flips.append(labels.nunique() > 1)
    return float(np.mean(agreements)), float(np.mean(flips))


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (evaluator, judge), group in runs.groupby(["evaluator", "judge"], sort=True):
        ci_low, ci_high = bootstrap_accuracy(group)
        precision, recall, f1, kappa = classification_metrics(group)
        consistency, flip_rate = consistency_metrics(group)
        entropy = float(group.groupby("example_id")["label"].apply(label_entropy_bits).mean())
        latency = pd.to_numeric(group["latency_ms"], errors="coerce")
        costs = pd.to_numeric(group["cost_usd"], errors="coerce")
        cost_deltas = pd.to_numeric(group["cost_delta_usd"], errors="coerce")
        jev_rows = group[pd.to_numeric(group["p_label"], errors="coerce").notna()]
        brier = math.nan
        if not jev_rows.empty:
            probabilities = pd.to_numeric(jev_rows["p_label"], errors="coerce")
            brier = float(np.mean((probabilities - jev_rows["correct"].astype(float)) ** 2))
        rows.append(
            {
                "evaluator": evaluator,
                "judge": judge,
                "runs": len(group),
                "accuracy": float(group["correct"].mean()),
                "accuracy_ci_low": ci_low,
                "accuracy_ci_high": ci_high,
                "macro_precision": precision,
                "macro_recall": recall,
                "macro_f1": f1,
                "cohen_kappa": kappa,
                "consistency": consistency,
                "flip_rate": flip_rate,
                "mean_label_entropy_bits": entropy,
                "median_latency_ms": float(latency.median()),
                "p95_latency_ms": float(latency.quantile(0.95)),
                "mean_prompt_tokens": float(
                    pd.to_numeric(group["prompt_tokens"], errors="coerce").mean()
                ),
                "mean_completion_tokens": float(
                    pd.to_numeric(group["completion_tokens"], errors="coerce").mean()
                ),
                "cost_per_100_evaluations_usd": float(costs.mean() * 100),
                "failures": int(group["failure_class"].notna().sum()),
                "missing_costs": int(costs.isna().sum()),
                "cost_disagreements": int((cost_deltas.abs() > 1e-8).sum()),
                "top_label_brier": brier,
            }
        )
    return pd.DataFrame(rows)


def get_example_results(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (evaluator, judge, example_id), group in runs.groupby(["evaluator", "judge", "example_id"]):
        labels = group["label"].fillna("<failure>").astype(str)
        counts = labels.value_counts()
        majority_label = sorted(counts[counts == counts.max()].index)[0]
        expected = str(group["expected"].iloc[0])
        rows.append(
            {
                "evaluator": evaluator,
                "judge": judge,
                "example_id": example_id,
                "expected": expected,
                "majority_label": majority_label,
                "majority_correct": majority_label == expected,
                "mean_correct": float(group["correct"].mean()),
                "label_entropy_bits": label_entropy_bits(labels),
            }
        )
    return pd.DataFrame(rows)


def get_agreement_matrix(
    *, example_results: pd.DataFrame, evaluator: str | None = None
) -> tuple[list[str], np.ndarray]:
    rows = example_results
    index = ["evaluator", "example_id"]
    if evaluator is not None:
        rows = rows[rows["evaluator"] == evaluator]
        index = ["example_id"]
    pivot = rows.pivot(index=index, columns="judge", values="majority_label")
    judges = [judge for judge in JUDGE_ORDER if judge in pivot.columns]
    matrix = np.eye(len(judges))
    for left_index, left in enumerate(judges):
        for right_index, right in enumerate(judges):
            if left_index == right_index:
                continue
            shared = pivot[[left, right]].dropna()
            matrix[left_index, right_index] = (
                float((shared[left] == shared[right]).mean()) if len(shared) else math.nan
            )
    return judges, matrix


def adjust_p_values(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    if not len(values):
        return values
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running_minimum = 1.0
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = len(values) - reverse_rank + 1
        running_minimum = min(running_minimum, values[index] * len(values) / rank)
        adjusted[index] = min(running_minimum, 1.0)
    return adjusted


def bootstrap_mean_interval(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 2 or np.all(values == values[0]):
        mean = float(values.mean())
        return mean, mean
    interval = bootstrap(
        (values,),
        np.mean,
        confidence_level=0.95,
        n_resamples=5_000,
        method="percentile",
        rng=np.random.default_rng(20260921),
    ).confidence_interval
    return float(interval.low), float(interval.high)


def mcnemar_exact(left: pd.Series, right: pd.Series) -> tuple[int, int, float]:
    left_wins = int((left & ~right).sum())
    right_wins = int((~left & right).sum())
    discordant = left_wins + right_wins
    p_value = (
        float(binomtest(min(left_wins, right_wins), discordant, 0.5).pvalue) if discordant else 1.0
    )
    return left_wins, right_wins, p_value


def cochran_q_test(outcomes: pd.DataFrame) -> tuple[float, float]:
    matrix = outcomes.to_numpy(dtype=float)
    judge_totals = matrix.sum(axis=0)
    example_totals = matrix.sum(axis=1)
    total = float(judge_totals.sum())
    judges = matrix.shape[1]
    denominator = judges * total - float(np.square(example_totals).sum())
    if denominator == 0:
        return 0.0, 1.0
    statistic = (
        (judges - 1) * (judges * float(np.square(judge_totals).sum()) - total**2) / denominator
    )
    return statistic, float(chi2.sf(statistic, judges - 1))


def compare_jev_modes(example_results: pd.DataFrame) -> pd.DataFrame:
    jev = example_results[example_results["judge"].isin(["jev-slot-in", "jev-native"])]
    accuracy = jev.pivot(index=["evaluator", "example_id"], columns="judge", values="mean_correct")
    majority = jev.pivot(
        index=["evaluator", "example_id"], columns="judge", values="majority_correct"
    )
    rows: list[dict[str, Any]] = []
    for evaluator in sorted(accuracy.index.get_level_values("evaluator").unique()):
        evaluator_accuracy = accuracy.loc[evaluator].dropna()
        evaluator_majority = majority.loc[evaluator].dropna()
        differences = (
            evaluator_accuracy["jev-native"] - evaluator_accuracy["jev-slot-in"]
        ).to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_mean_interval(differences)
        native_wins, slot_in_wins, p_value = mcnemar_exact(
            evaluator_majority["jev-native"].astype(bool),
            evaluator_majority["jev-slot-in"].astype(bool),
        )
        rows.append(
            {
                "evaluator": evaluator,
                "examples": len(evaluator_accuracy),
                "slot_in_accuracy": float(evaluator_accuracy["jev-slot-in"].mean()),
                "native_accuracy": float(evaluator_accuracy["jev-native"].mean()),
                "native_minus_slot_in": float(differences.mean()),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "native_only_correct": native_wins,
                "slot_in_only_correct": slot_in_wins,
                "mcnemar_p_value": p_value,
            }
        )
    comparison = pd.DataFrame(rows)
    comparison["mcnemar_q_value"] = adjust_p_values(comparison["mcnemar_p_value"])
    return comparison


def analyze_task_divergence(
    *, example_results: pd.DataFrame, summary: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for evaluator, evaluator_rows in example_results.groupby("evaluator"):
        labels = evaluator_rows.pivot(
            index="example_id", columns="judge", values="majority_label"
        ).dropna()
        outcomes = evaluator_rows.pivot(
            index="example_id", columns="judge", values="majority_correct"
        ).dropna()
        model_entropy = labels.apply(label_entropy_bits, axis=1)
        q_statistic, p_value = cochran_q_test(outcomes)
        accuracies = summary[summary["evaluator"] == evaluator].set_index("judge")["accuracy"]
        rows.append(
            {
                "evaluator": evaluator,
                "examples": len(labels),
                "accuracy_min": float(accuracies.min()),
                "accuracy_max": float(accuracies.max()),
                "accuracy_range": float(accuracies.max() - accuracies.min()),
                "best_judge": str(accuracies.idxmax()),
                "worst_judge": str(accuracies.idxmin()),
                "mean_cross_model_entropy_bits": float(model_entropy.mean()),
                "unanimous_examples": float((labels.nunique(axis=1) == 1).mean()),
                "cochran_q_statistic": q_statistic,
                "cochran_q_p_value": p_value,
            }
        )
    divergence = pd.DataFrame(rows)
    divergence["cochran_q_q_value"] = adjust_p_values(divergence["cochran_q_p_value"])
    return divergence


def compare_all_judges(example_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    judges = [judge for judge in JUDGE_ORDER if judge in set(example_results["judge"])]
    for evaluator, evaluator_rows in example_results.groupby("evaluator"):
        outcomes = evaluator_rows.pivot(
            index="example_id", columns="judge", values="majority_correct"
        )
        for left, right in combinations(judges, 2):
            paired = outcomes[[left, right]].dropna().astype(bool)
            left_wins, right_wins, p_value = mcnemar_exact(paired[left], paired[right])
            rows.append(
                {
                    "evaluator": evaluator,
                    "left_judge": left,
                    "right_judge": right,
                    "examples": len(paired),
                    "left_accuracy": float(paired[left].mean()),
                    "right_accuracy": float(paired[right].mean()),
                    "accuracy_difference": float(paired[left].mean() - paired[right].mean()),
                    "left_only_correct": left_wins,
                    "right_only_correct": right_wins,
                    "mcnemar_p_value": p_value,
                }
            )
    comparisons = pd.DataFrame(rows)
    comparisons["mcnemar_q_value"] = adjust_p_values(comparisons["mcnemar_p_value"])
    return comparisons


def get_jev_probability_flip_rows(runs: pd.DataFrame) -> pd.DataFrame:
    conventional = runs[~runs["judge"].isin(["jev-slot-in", "jev-native"])]
    conventional_flips = (
        conventional.assign(label=conventional["label"].fillna("<failure>").astype(str))
        .groupby(["evaluator", "example_id", "judge"])["label"]
        .nunique()
        .gt(1)
        .groupby(["evaluator", "example_id"])
        .mean()
        .rename("other_judge_flip_rate")
        .reset_index()
    )
    jev = runs[
        runs["judge"].isin(["jev-slot-in", "jev-native"])
        & pd.to_numeric(runs["p_label"], errors="coerce").notna()
    ].copy()
    jev["p_label"] = pd.to_numeric(jev["p_label"], errors="coerce")
    jev_probabilities = (
        jev.groupby(["evaluator", "judge", "example_id"])["p_label"]
        .mean()
        .rename("mean_p_label")
        .reset_index()
    )
    return jev_probabilities.merge(
        conventional_flips,
        on=["evaluator", "example_id"],
        how="inner",
    )


def binary_entropy_bits(probabilities: pd.Series) -> pd.Series:
    values = pd.to_numeric(probabilities, errors="coerce").to_numpy(dtype=float)
    entropy = np.zeros(len(values), dtype=float)
    valid = (values > 0) & (values < 1)
    entropy[valid] = -(
        values[valid] * np.log2(values[valid]) + (1 - values[valid]) * np.log2(1 - values[valid])
    )
    entropy[~np.isfinite(values)] = np.nan
    return pd.Series(entropy, index=probabilities.index)


def bootstrap_spearman_interval(
    left: np.ndarray, right: np.ndarray, *, n_resamples: int = 5_000
) -> tuple[float, float]:
    rng = np.random.default_rng(20260921)
    correlations: list[float] = []
    for _ in range(n_resamples):
        indices = rng.integers(0, len(left), len(left))
        correlation = spearmanr(left[indices], right[indices]).statistic
        if np.isfinite(correlation):
            correlations.append(float(correlation))
    low, high = np.quantile(correlations, [0.025, 0.975])
    return float(low), float(high)


def task_stratified_permutation_p_value(
    frame: pd.DataFrame,
    *,
    left: str,
    right: str,
    n_resamples: int = 5_000,
) -> float:
    left_values = frame[left].to_numpy(dtype=float)
    right_values = frame[right].to_numpy(dtype=float)
    observed = abs(float(spearmanr(left_values, right_values).statistic))
    task_indices = [
        indices.to_numpy(dtype=int)
        for _, indices in frame.reset_index(drop=True).groupby("evaluator").groups.items()
    ]
    rng = np.random.default_rng(20260921)
    at_least_as_extreme = 0
    for _ in range(n_resamples):
        shuffled = left_values.copy()
        for indices in task_indices:
            shuffled[indices] = rng.permutation(shuffled[indices])
        correlation = abs(float(spearmanr(shuffled, right_values).statistic))
        at_least_as_extreme += correlation >= observed
    return (at_least_as_extreme + 1) / (n_resamples + 1)


def binary_rank_auc(scores: pd.Series, outcomes: pd.Series) -> float:
    positive = outcomes.astype(bool).to_numpy()
    ranks = rankdata(scores.to_numpy(dtype=float))
    positive_count = int(positive.sum())
    negative_count = len(positive) - positive_count
    if positive_count == 0 or negative_count == 0:
        return math.nan
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - positive_count * (positive_count + 1) / 2) / (
        positive_count * negative_count
    )


def analyze_jev_uncertainty_alignment(
    runs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    conventional = runs[~runs["judge"].isin(["jev-slot-in", "jev-native"])]
    llm_entropy = (
        conventional.groupby(["evaluator", "example_id", "judge"])["label"]
        .apply(label_entropy_bits)
        .rename("judge_label_entropy_bits")
        .reset_index()
    )
    llm_examples = (
        llm_entropy.groupby(["evaluator", "example_id"])
        .agg(
            llm_mean_label_entropy_bits=("judge_label_entropy_bits", "mean"),
            llm_judges_with_flips=(
                "judge_label_entropy_bits",
                lambda values: int((values > 0).sum()),
            ),
        )
        .reset_index()
    )
    llm_examples["llm_any_flip"] = llm_examples["llm_judges_with_flips"] > 0

    jev = runs[runs["judge"].isin(["jev-slot-in", "jev-native"])].copy()
    jev["p_label"] = pd.to_numeric(jev["p_label"], errors="coerce")
    jev = jev[jev["p_label"].notna()].copy()
    label_counts = jev.groupby("evaluator")["label"].nunique()
    if (label_counts > 2).any():
        evaluators = ", ".join(label_counts[label_counts > 2].index)
        raise RuntimeError(f"Jev uncertainty analysis requires binary labels: {evaluators}")
    jev["predictive_entropy_bits"] = binary_entropy_bits(jev["p_label"])
    jev["reference_label_probability"] = np.where(
        jev["label"] == jev["expected"], jev["p_label"], 1 - jev["p_label"]
    )
    jev_examples = (
        jev.groupby(["evaluator", "example_id", "judge"])
        .agg(
            jev_mean_predictive_entropy_bits=("predictive_entropy_bits", "mean"),
            jev_reference_probability_sd=(
                "reference_label_probability",
                lambda values: float(np.std(values, ddof=0)),
            ),
            jev_mean_top_label_probability=("p_label", "mean"),
        )
        .reset_index()
        .rename(columns={"judge": "jev_mode"})
    )
    alignment = jev_examples.merge(llm_examples, on=["evaluator", "example_id"], how="inner")

    measures = [
        ("predictive_entropy", "jev_mean_predictive_entropy_bits"),
        ("probability_noise", "jev_reference_probability_sd"),
    ]
    overall_rows: list[dict[str, Any]] = []
    for mode, mode_rows in alignment.groupby("jev_mode"):
        for measure_name, measure in measures:
            left = mode_rows[measure].to_numpy(dtype=float)
            right = mode_rows["llm_mean_label_entropy_bits"].to_numpy(dtype=float)
            correlation = float(spearmanr(left, right).statistic)
            ci_low, ci_high = bootstrap_spearman_interval(left, right)
            p_value = task_stratified_permutation_p_value(
                mode_rows,
                left=measure,
                right="llm_mean_label_entropy_bits",
            )
            stable = mode_rows[~mode_rows["llm_any_flip"]]
            variable = mode_rows[mode_rows["llm_any_flip"]]
            overall_rows.append(
                {
                    "jev_mode": mode,
                    "measure": measure_name,
                    "examples": len(mode_rows),
                    "spearman_rho": correlation,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "task_stratified_permutation_p_value": p_value,
                    "stable_examples_mean": float(stable[measure].mean()),
                    "variable_examples_mean": float(variable[measure].mean()),
                    "stable_examples": len(stable),
                    "variable_examples": len(variable),
                    "any_flip_auc": binary_rank_auc(mode_rows[measure], mode_rows["llm_any_flip"]),
                }
            )
    overall = pd.DataFrame(overall_rows)
    overall["bh_q_value"] = adjust_p_values(overall["task_stratified_permutation_p_value"])

    task_rows: list[dict[str, Any]] = []
    for (mode, evaluator), group in alignment.groupby(["jev_mode", "evaluator"]):
        left = group["jev_mean_predictive_entropy_bits"]
        right = group["llm_mean_label_entropy_bits"]
        correlation = (
            float(spearmanr(left, right).statistic)
            if left.nunique() > 1 and right.nunique() > 1
            else math.nan
        )
        p_value = float(spearmanr(left, right).pvalue) if np.isfinite(correlation) else 1.0
        task_rows.append(
            {
                "jev_mode": mode,
                "evaluator": evaluator,
                "examples": len(group),
                "spearman_rho": correlation,
                "p_value": p_value,
            }
        )
    by_task = pd.DataFrame(task_rows)
    by_task["bh_q_value"] = adjust_p_values(by_task["p_value"])
    return alignment, overall, by_task


def write_figure(figure: go.Figure, *, path: Path) -> None:
    figure.update_layout(template="plotly_white")
    figure.write_image(path, width=1600, height=900, scale=1.5)


def create_figures(
    *,
    runs: pd.DataFrame,
    summary: pd.DataFrame,
    example_results: pd.DataFrame,
    jev_comparison: pd.DataFrame,
    jev_uncertainty_alignment: pd.DataFrame,
    jev_uncertainty_tests: pd.DataFrame,
    jev_uncertainty_by_task: pd.DataFrame,
    task_divergence: pd.DataFrame,
    pairwise_significance: pd.DataFrame,
    figures_directory: Path,
) -> list[tuple[str, str, go.Figure]]:
    figures_directory.mkdir(parents=True, exist_ok=True)
    figures: list[tuple[str, str, go.Figure]] = []
    judge_order = [judge for judge in JUDGE_ORDER if judge in set(summary["judge"])]
    evaluator_order = task_divergence.sort_values("accuracy_range", ascending=False)[
        "evaluator"
    ].tolist()

    accuracy_table = (
        summary.pivot(index="evaluator", columns="judge", values="accuracy").reindex(
            index=evaluator_order, columns=judge_order
        )
        * 100
    )
    accuracy = go.Figure(
        go.Heatmap(
            z=accuracy_table.to_numpy(),
            x=accuracy_table.columns,
            y=accuracy_table.index,
            zmin=float(accuracy_table.min().min()),
            zmax=100,
            colorscale="Blues",
            text=np.char.add(np.round(accuracy_table.to_numpy(), 1).astype(str), "%"),
            texttemplate="%{text}",
            hovertemplate="%{y}<br>%{x}: %{z:.1f}%<extra></extra>",
        )
    )
    accuracy.update_layout(title="Accuracy by task and judge")
    accuracy.update_yaxes(autorange="reversed")
    write_figure(accuracy, path=figures_directory / "accuracy-heatmap.png")
    figures.append(("Model and task differences", "Accuracy at a glance", accuracy))

    macro_f1_table = (
        summary.pivot(index="evaluator", columns="judge", values="macro_f1").reindex(
            index=evaluator_order, columns=judge_order
        )
        * 100
    )
    macro_f1_values = macro_f1_table.to_numpy()
    macro_f1_text = np.array(
        [["N/A" if pd.isna(value) else f"{value:.1f}%" for value in row] for row in macro_f1_values]
    )
    macro_f1 = go.Figure(
        go.Heatmap(
            z=macro_f1_values,
            x=macro_f1_table.columns,
            y=macro_f1_table.index,
            zmin=float(np.nanmin(macro_f1_values)),
            zmax=100,
            colorscale="Blues",
            text=macro_f1_text,
            texttemplate="%{text}",
            hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>",
        )
    )
    macro_f1.update_layout(title="Macro F1 by task and judge")
    macro_f1.update_yaxes(autorange="reversed")
    write_figure(macro_f1, path=figures_directory / "macro-f1-heatmap.png")
    figures.append(("Model and task differences", "Class-balanced performance", macro_f1))

    centered_accuracy = accuracy_table.sub(accuracy_table.mean(axis=1), axis=0)
    centered_limit = float(np.abs(centered_accuracy.to_numpy()).max())
    relative_accuracy = go.Figure(
        go.Heatmap(
            z=centered_accuracy.to_numpy(),
            x=centered_accuracy.columns,
            y=centered_accuracy.index,
            zmin=-centered_limit,
            zmax=centered_limit,
            zmid=0,
            colorscale="RdBu",
            text=np.char.add(
                np.where(centered_accuracy.to_numpy() >= 0, "+", ""),
                np.round(centered_accuracy.to_numpy(), 1).astype(str),
            ),
            texttemplate="%{text} pp",
            hovertemplate="%{y}<br>%{x}: %{z:+.1f} points from task mean<extra></extra>",
        )
    )
    relative_accuracy.update_layout(title="Accuracy relative to each task's judge mean")
    relative_accuracy.update_yaxes(autorange="reversed")
    write_figure(relative_accuracy, path=figures_directory / "relative-accuracy.png")
    figures.append(
        ("Model and task differences", "Relative strengths and weaknesses", relative_accuracy)
    )

    divergence_rows = task_divergence.sort_values("accuracy_range", ascending=False).copy()
    divergence_rows["label"] = divergence_rows.apply(
        lambda row: (
            f"{row['accuracy_range']:.1%} gap · "
            f"{row['mean_cross_model_entropy_bits']:.3f} bits"
            + (" · q<0.05" if row["cochran_q_q_value"] < 0.05 else "")
        ),
        axis=1,
    )
    divergence = px.bar(
        divergence_rows,
        x="accuracy_range",
        y="evaluator",
        orientation="h",
        color="mean_cross_model_entropy_bits",
        text="label",
        hover_data=[
            "best_judge",
            "worst_judge",
            "unanimous_examples",
            "cochran_q_q_value",
        ],
        labels={
            "accuracy_range": "Best-to-worst accuracy gap",
            "mean_cross_model_entropy_bits": "Cross-model entropy (bits)",
        },
        title="Tasks where judge choice matters most",
        color_continuous_scale="Oranges",
    )
    divergence.update_traces(textposition="outside", cliponaxis=False)
    divergence.update_xaxes(tickformat=".0%")
    divergence.update_yaxes(autorange="reversed", title=None)
    write_figure(divergence, path=figures_directory / "task-divergence.png")
    figures.append(("Model and task differences", "Task divergence", divergence))

    judges, agreement_matrix = get_agreement_matrix(example_results=example_results)
    agreement = go.Figure(
        go.Heatmap(
            z=agreement_matrix,
            x=judges,
            y=judges,
            zmin=0.5,
            zmax=1,
            colorscale="Blues",
            text=np.round(agreement_matrix, 3),
            texttemplate="%{text}",
            hovertemplate="%{y} vs %{x}: %{z:.1%} majority-label agreement<extra></extra>",
        )
    )
    agreement.update_layout(title="Pairwise majority-label agreement across all tasks")
    write_figure(agreement, path=figures_directory / "overall-agreement.png")
    figures.append(("Model and task differences", "Overall judge agreement", agreement))

    significant_wins = pd.DataFrame(0, index=judge_order, columns=judge_order, dtype=int)
    for row in pairwise_significance.itertuples():
        if row.mcnemar_q_value >= 0.05 or row.accuracy_difference == 0:
            continue
        winner = row.left_judge if row.accuracy_difference > 0 else row.right_judge
        loser = row.right_judge if row.accuracy_difference > 0 else row.left_judge
        significant_wins.loc[winner, loser] += 1
    win_matrix = go.Figure(
        go.Heatmap(
            z=significant_wins.to_numpy(),
            x=significant_wins.columns,
            y=significant_wins.index,
            zmin=0,
            zmax=max(1, int(significant_wins.to_numpy().max())),
            colorscale="Blues",
            text=significant_wins.to_numpy(),
            texttemplate="%{text}",
            hovertemplate="%{y} significantly beats %{x} on %{z} tasks<extra></extra>",
        )
    )
    win_matrix.update_layout(
        title="Task-level significant wins after paired tests and BH correction",
        xaxis_title="Losing judge",
        yaxis_title="Winning judge",
    )
    write_figure(win_matrix, path=figures_directory / "significant-wins.png")
    figures.append(("Model and task differences", "Significant pairwise wins", win_matrix))

    jev_plot = jev_comparison.sort_values("native_minus_slot_in")
    jev_effect = go.Figure(
        go.Scatter(
            x=jev_plot["native_minus_slot_in"] * 100,
            y=jev_plot["evaluator"],
            mode="markers+text",
            text=np.where(jev_plot["mcnemar_q_value"] < 0.05, "q<0.05", ""),
            textposition="middle right",
            error_x={
                "type": "data",
                "symmetric": False,
                "array": (jev_plot["ci_high"] - jev_plot["native_minus_slot_in"]) * 100,
                "arrayminus": (jev_plot["native_minus_slot_in"] - jev_plot["ci_low"]) * 100,
            },
            customdata=jev_plot[
                [
                    "slot_in_accuracy",
                    "native_accuracy",
                    "mcnemar_p_value",
                    "mcnemar_q_value",
                ]
            ],
            hovertemplate=(
                "%{y}<br>native - slot-in: %{x:+.1f} pp"
                "<br>slot-in: %{customdata[0]:.1%}"
                "<br>native: %{customdata[1]:.1%}"
                "<br>McNemar p=%{customdata[2]:.3g}, q=%{customdata[3]:.3g}<extra></extra>"
            ),
        )
    )
    jev_effect.add_vline(x=0, line_dash="dash", line_color="gray")
    jev_effect.update_layout(
        title="Jev native accuracy minus slot-in accuracy",
        xaxis_title="Accuracy difference (percentage points, 95% paired bootstrap CI)",
        yaxis_title="Evaluator",
    )
    write_figure(jev_effect, path=figures_directory / "jev-paired-effects.png")
    figures.append(("Jev request shapes", "Jev native versus slot-in", jev_effect))

    uncertainty_scatter = px.scatter(
        jev_uncertainty_alignment,
        x="jev_mean_predictive_entropy_bits",
        y="llm_mean_label_entropy_bits",
        color="evaluator",
        facet_col="jev_mode",
        category_orders={"jev_mode": ["jev-slot-in", "jev-native"]},
        opacity=0.6,
        hover_data=["example_id", "llm_judges_with_flips"],
        labels={
            "jev_mean_predictive_entropy_bits": "Jev predictive entropy (bits)",
            "llm_mean_label_entropy_bits": "Mean LLM repetition entropy (bits)",
            "evaluator": "Task",
            "jev_mode": "Jev mode",
        },
        title="Jev uncertainty versus conventional-LLM response instability",
    )
    uncertainty_scatter.update_xaxes(range=[-0.02, 1.02])
    uncertainty_scatter.update_yaxes(range=[-0.02, 1.02])
    for index, mode in enumerate(["jev-slot-in", "jev-native"]):
        result = jev_uncertainty_tests[
            (jev_uncertainty_tests["jev_mode"] == mode)
            & (jev_uncertainty_tests["measure"] == "predictive_entropy")
        ].iloc[0]
        axis_suffix = "" if index == 0 else "2"
        uncertainty_scatter.add_annotation(
            x=0.02,
            y=0.98,
            xref=f"x{axis_suffix} domain",
            yref=f"y{axis_suffix} domain",
            text=(
                f"ρ={result['spearman_rho']:.3f}<br>"
                f"q={result['bh_q_value']:.4f}<br>"
                f"any-flip AUC={result['any_flip_auc']:.3f}"
            ),
            showarrow=False,
            align="left",
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
        )
    write_figure(
        uncertainty_scatter,
        path=figures_directory / "jev-uncertainty-vs-llm-entropy.png",
    )
    figures.append(
        (
            "Jev probability alignment",
            "Predictive uncertainty versus LLM response entropy",
            uncertainty_scatter,
        )
    )

    probability_noise = px.scatter(
        jev_uncertainty_alignment,
        x="jev_reference_probability_sd",
        y="llm_mean_label_entropy_bits",
        color="evaluator",
        facet_col="jev_mode",
        category_orders={"jev_mode": ["jev-slot-in", "jev-native"]},
        opacity=0.6,
        hover_data=["example_id", "llm_judges_with_flips"],
        labels={
            "jev_reference_probability_sd": "SD of Jev reference-label probability",
            "llm_mean_label_entropy_bits": "Mean LLM repetition entropy (bits)",
            "evaluator": "Task",
            "jev_mode": "Jev mode",
        },
        title="Probability noise versus conventional-LLM response instability",
    )
    probability_noise.update_yaxes(range=[-0.02, 1.02])
    for index, mode in enumerate(["jev-slot-in", "jev-native"]):
        result = jev_uncertainty_tests[
            (jev_uncertainty_tests["jev_mode"] == mode)
            & (jev_uncertainty_tests["measure"] == "probability_noise")
        ].iloc[0]
        axis_suffix = "" if index == 0 else "2"
        probability_noise.add_annotation(
            x=0.98,
            y=0.98,
            xref=f"x{axis_suffix} domain",
            yref=f"y{axis_suffix} domain",
            text=(
                f"ρ={result['spearman_rho']:.3f}<br>"
                f"q={result['bh_q_value']:.4f}<br>"
                f"any-flip AUC={result['any_flip_auc']:.3f}"
            ),
            showarrow=False,
            align="right",
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
        )
    write_figure(
        probability_noise,
        path=figures_directory / "jev-probability-noise-vs-llm-entropy.png",
    )
    figures.append(
        (
            "Jev probability alignment",
            "Probability variation across repetitions",
            probability_noise,
        )
    )

    task_alignment = jev_uncertainty_by_task.pivot(
        index="evaluator", columns="jev_mode", values="spearman_rho"
    ).reindex(index=evaluator_order, columns=["jev-slot-in", "jev-native"])
    task_alignment_q = jev_uncertainty_by_task.pivot(
        index="evaluator", columns="jev_mode", values="bh_q_value"
    ).reindex(index=evaluator_order, columns=["jev-slot-in", "jev-native"])
    task_alignment_text = np.empty(task_alignment.shape, dtype=object)
    for row_index in range(task_alignment.shape[0]):
        for column_index in range(task_alignment.shape[1]):
            value = task_alignment.iat[row_index, column_index]
            q_value = task_alignment_q.iat[row_index, column_index]
            task_alignment_text[row_index, column_index] = (
                "n/a"
                if not np.isfinite(value)
                else f"{value:.2f}" + (" *" if q_value < 0.05 else "")
            )
    task_alignment_figure = go.Figure(
        go.Heatmap(
            z=task_alignment.to_numpy(),
            x=task_alignment.columns,
            y=task_alignment.index,
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale="RdBu",
            text=task_alignment_text,
            texttemplate="%{text}",
            customdata=task_alignment_q.to_numpy(),
            hovertemplate=(
                "%{y}<br>%{x}: Spearman ρ=%{z:.3f}<br>BH q=%{customdata:.3g}<extra></extra>"
            ),
        )
    )
    task_alignment_figure.update_layout(
        title="Within-task alignment between Jev uncertainty and LLM entropy (* q<0.05)",
        xaxis_title="Jev mode",
    )
    task_alignment_figure.update_yaxes(autorange="reversed")
    write_figure(
        task_alignment_figure,
        path=figures_directory / "jev-uncertainty-task-correlations.png",
    )
    figures.append(
        (
            "Jev probability alignment",
            "Alignment by eval task",
            task_alignment_figure,
        )
    )

    entropy_table = summary.pivot(
        index="evaluator", columns="judge", values="mean_label_entropy_bits"
    ).reindex(index=evaluator_order, columns=judge_order)
    entropy = go.Figure(
        go.Heatmap(
            z=entropy_table.to_numpy(),
            x=entropy_table.columns,
            y=entropy_table.index,
            zmin=0,
            zmax=max(0.01, float(entropy_table.max().max())),
            colorscale="Oranges",
            text=np.round(entropy_table.to_numpy(), 3),
            texttemplate="%{text}",
            hovertemplate="%{y}<br>%{x}: %{z:.3f} bits<extra></extra>",
        )
    )
    entropy.update_layout(title="Run-to-run label entropy across ten repetitions")
    entropy.update_yaxes(autorange="reversed")
    write_figure(entropy, path=figures_directory / "repeatability-entropy.png")
    figures.append(("Variance and uncertainty", "Repeatability entropy", entropy))

    probability_flip_rows = get_jev_probability_flip_rows(runs)
    if not probability_flip_rows.empty:
        probability_flip_rows["probability_bin"] = pd.cut(
            probability_flip_rows["mean_p_label"],
            bins=np.linspace(0, 1, 11),
            include_lowest=True,
        )
        probability_flip_summary = (
            probability_flip_rows.groupby(["evaluator", "judge", "probability_bin"], observed=True)
            .agg(
                mean_p_label=("mean_p_label", "mean"),
                other_judge_flip_rate=("other_judge_flip_rate", "mean"),
                examples=("example_id", "size"),
            )
            .reset_index()
        )
        probability_flips = px.line(
            probability_flip_summary,
            x="mean_p_label",
            y="other_judge_flip_rate",
            color="judge",
            facet_col="evaluator",
            facet_col_wrap=2,
            markers=True,
            hover_data=["examples"],
            title="Jev top-label probability versus conventional-judge flip rate",
        )
        write_figure(
            probability_flips,
            path=figures_directory / "jev-probability-vs-other-judge-flips.png",
        )
        figures.append(
            (
                "Jev request shapes",
                "Jev probability versus conventional-judge flips",
                probability_flips,
            )
        )

    reliability_rows = runs[
        runs["judge"].isin(["jev-slot-in", "jev-native"])
        & pd.to_numeric(runs["p_label"], errors="coerce").notna()
    ].copy()
    if not reliability_rows.empty:
        reliability_rows["p_label"] = pd.to_numeric(reliability_rows["p_label"], errors="coerce")
        reliability_rows["probability_bin"] = pd.cut(
            reliability_rows["p_label"],
            bins=np.linspace(0, 1, 11),
            include_lowest=True,
        )
        reliability = (
            reliability_rows.groupby(["judge", "probability_bin"], observed=True)
            .agg(
                predicted_probability=("p_label", "mean"),
                observed_accuracy=("correct", "mean"),
                count=("correct", "size"),
            )
            .reset_index()
        )
        reliability_figure = px.line(
            reliability,
            x="predicted_probability",
            y="observed_accuracy",
            color="judge",
            markers=True,
            hover_data=["count"],
            title="Jev top-label reliability",
        )
        reliability_figure.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line={"dash": "dash", "color": "gray"},
                name="perfect calibration",
            )
        )
        write_figure(
            reliability_figure,
            path=figures_directory / "jev-reliability.png",
        )
        figures.append(
            ("Jev request shapes", "Jev reliability and calibration", reliability_figure)
        )

    judge_overview = (
        runs.groupby("judge")
        .agg(accuracy=("correct", "mean"), cost_per_run=("cost_usd", "mean"))
        .reset_index()
    )
    cost_accuracy = px.scatter(
        judge_overview,
        x="cost_per_run",
        y="accuracy",
        color="judge",
        text="judge",
        log_x=True,
        title="Overall cost versus accuracy",
        labels={"cost_per_run": "Mean cost per evaluation (USD)", "accuracy": "Accuracy"},
    )
    cost_accuracy.update_traces(textposition="top center")
    cost_accuracy.update_yaxes(tickformat=".0%")
    write_figure(cost_accuracy, path=figures_directory / "cost-vs-accuracy.png")
    figures.append(("Variance and uncertainty", "Cost versus accuracy", cost_accuracy))

    latency = px.box(
        runs,
        x="judge",
        y="latency_ms",
        color="judge",
        points=False,
        log_y=True,
        category_orders={"judge": judge_order},
        title="Run latency by judge",
    )
    write_figure(latency, path=figures_directory / "latency.png")
    figures.append(("Variance and uncertainty", "Latency distributions", latency))

    return figures


def build_findings(
    *,
    summary: pd.DataFrame,
    jev_comparison: pd.DataFrame,
    jev_uncertainty_tests: pd.DataFrame,
    task_divergence: pd.DataFrame,
    pairwise_significance: pd.DataFrame,
) -> list[str]:
    most_divergent = task_divergence.sort_values("accuracy_range", ascending=False).iloc[0]
    jev_largest = jev_comparison.iloc[jev_comparison["native_minus_slot_in"].abs().argmax()]
    most_variable = summary.sort_values("mean_label_entropy_bits", ascending=False).iloc[0]
    significant_tasks = task_divergence[task_divergence["cochran_q_q_value"] < 0.05]
    significant_pairs = pairwise_significance[pairwise_significance["mcnemar_q_value"] < 0.05]
    jev_significant = jev_comparison[jev_comparison["mcnemar_q_value"] < 0.05]
    predictive_alignment = jev_uncertainty_tests[
        jev_uncertainty_tests["measure"] == "predictive_entropy"
    ].set_index("jev_mode")
    slot_in_alignment = predictive_alignment.loc["jev-slot-in"]
    native_alignment = predictive_alignment.loc["jev-native"]
    slot_in_variable_mean = slot_in_alignment["variable_examples_mean"]
    slot_in_stable_mean = slot_in_alignment["stable_examples_mean"]
    return [
        (
            f"{most_divergent['evaluator']} has the widest model accuracy range at "
            f"{most_divergent['accuracy_range']:.1%}. "
            f"{most_divergent['best_judge']} leads and {most_divergent['worst_judge']} trails."
        ),
        (
            f"{len(significant_tasks)} of {len(task_divergence)} tasks show a judge effect under "
            "Cochran's Q test after Benjamini-Hochberg correction. "
            f"{len(significant_pairs)} of {len(pairwise_significance)} paired task comparisons "
            "remain significant after the same correction."
        ),
        (
            f"The largest Jev request-shape effect is on {jev_largest['evaluator']}: native "
            f"minus slot-in is {jev_largest['native_minus_slot_in']:+.1%}. "
            f"{len(jev_significant)} of {len(jev_comparison)} Jev task comparisons have q < 0.05."
        ),
        (
            "Jev uncertainty lines up with conventional-LLM response instability. "
            f"Predictive entropy has Spearman rho {slot_in_alignment['spearman_rho']:.3f} "
            f"for slot-in and {native_alignment['spearman_rho']:.3f} for native; both "
            "remain significant when probabilities are shuffled only within eval tasks. "
            f"Examples with an LLM label flip average {slot_in_variable_mean:.3f} "
            f"bits of slot-in uncertainty versus {slot_in_stable_mean:.3f} "
            "bits for stable examples."
        ),
        (
            f"The least repeatable judge/task cell is {most_variable['judge']} on "
            f"{most_variable['evaluator']} at {most_variable['mean_label_entropy_bits']:.3f} "
            "bits of mean label entropy. Zero bits means every repetition returned the same label."
        ),
    ]


def write_report(
    *,
    sweep_run_id: str,
    runs: pd.DataFrame,
    summary: pd.DataFrame,
    class_balance: pd.DataFrame,
    class_metrics: pd.DataFrame,
    jev_comparison: pd.DataFrame,
    jev_uncertainty_tests: pd.DataFrame,
    jev_uncertainty_by_task: pd.DataFrame,
    task_divergence: pd.DataFrame,
    pairwise_significance: pd.DataFrame,
    excluded_examples: pd.DataFrame,
    excluded_run_count: int,
    figures: Iterable[tuple[str, str, go.Figure]],
    output_path: Path,
) -> None:
    findings = build_findings(
        summary=summary,
        jev_comparison=jev_comparison,
        jev_uncertainty_tests=jev_uncertainty_tests,
        task_divergence=task_divergence,
        pairwise_significance=pairwise_significance,
    )
    class_balance_table = class_balance.rename(
        columns={
            "evaluator": "Task",
            "label": "Reference label",
            "base_examples": "Base examples",
            "expected_share": "Share",
            "expected_class_count": "Classes",
            "classification_metrics_available": "P/R/F1 available",
        }
    )
    class_balance_table["Share"] = class_balance_table["Share"].map(lambda value: f"{value:.1%}")
    class_balance_table["P/R/F1 available"] = class_balance_table["P/R/F1 available"].map(
        {True: "Yes", False: "No"}
    )
    class_metrics_table = class_metrics.rename(
        columns={
            "evaluator": "Task",
            "judge": "Judge",
            "label": "Reference label",
            "base_examples": "Base examples",
            "runs": "Runs",
            "expected_share": "Share",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "metrics_available": "Metrics available",
        }
    )
    for column in ["Share", "Precision", "Recall", "F1"]:
        class_metrics_table[column] = class_metrics_table[column].map(
            lambda value: "N/A" if pd.isna(value) else f"{value:.1%}"
        )
    class_metrics_table["Metrics available"] = class_metrics_table["Metrics available"].map(
        {True: "Yes", False: "No"}
    )
    task_table = task_divergence[
        [
            "evaluator",
            "accuracy_range",
            "mean_cross_model_entropy_bits",
            "unanimous_examples",
            "best_judge",
            "worst_judge",
            "cochran_q_p_value",
            "cochran_q_q_value",
        ]
    ].sort_values("accuracy_range", ascending=False)
    task_table = task_table.rename(
        columns={
            "evaluator": "Task",
            "accuracy_range": "Accuracy range",
            "mean_cross_model_entropy_bits": "Cross-model entropy (bits)",
            "unanimous_examples": "Unanimous examples",
            "best_judge": "Best judge",
            "worst_judge": "Worst judge",
            "cochran_q_p_value": "Cochran Q p",
            "cochran_q_q_value": "BH q",
        }
    )
    task_table["Accuracy range"] = task_table["Accuracy range"].map(lambda value: f"{value:.1%}")
    jev_table = jev_comparison[
        [
            "evaluator",
            "slot_in_accuracy",
            "native_accuracy",
            "native_minus_slot_in",
            "ci_low",
            "ci_high",
            "native_only_correct",
            "slot_in_only_correct",
            "mcnemar_p_value",
            "mcnemar_q_value",
        ]
    ].sort_values("native_minus_slot_in", ascending=False)
    jev_table = jev_table.rename(
        columns={
            "evaluator": "Task",
            "slot_in_accuracy": "Slot-in accuracy",
            "native_accuracy": "Native accuracy",
            "native_minus_slot_in": "Native - slot-in",
            "ci_low": "95% CI low",
            "ci_high": "95% CI high",
            "native_only_correct": "Native-only correct",
            "slot_in_only_correct": "Slot-in-only correct",
            "mcnemar_p_value": "McNemar p",
            "mcnemar_q_value": "BH q",
        }
    )
    for column in [
        "Slot-in accuracy",
        "Native accuracy",
        "Native - slot-in",
        "95% CI low",
        "95% CI high",
    ]:
        jev_table[column] = jev_table[column].map(lambda value: f"{value:.1%}")
    significant_pair_table = pairwise_significance[pairwise_significance["mcnemar_q_value"] < 0.05][
        [
            "evaluator",
            "left_judge",
            "right_judge",
            "left_accuracy",
            "right_accuracy",
            "accuracy_difference",
            "left_only_correct",
            "right_only_correct",
            "mcnemar_p_value",
            "mcnemar_q_value",
        ]
    ].sort_values("mcnemar_q_value")
    significant_pair_table = significant_pair_table.rename(
        columns={
            "evaluator": "Task",
            "left_judge": "Judge A",
            "right_judge": "Judge B",
            "left_accuracy": "A accuracy",
            "right_accuracy": "B accuracy",
            "accuracy_difference": "A - B",
            "left_only_correct": "A-only correct",
            "right_only_correct": "B-only correct",
            "mcnemar_p_value": "McNemar p",
            "mcnemar_q_value": "BH q",
        }
    )
    for column in ["A accuracy", "B accuracy", "A - B"]:
        significant_pair_table[column] = significant_pair_table[column].map(
            lambda value: f"{value:.1%}"
        )
    uncertainty_table = jev_uncertainty_tests[
        [
            "jev_mode",
            "measure",
            "examples",
            "spearman_rho",
            "ci_low",
            "ci_high",
            "task_stratified_permutation_p_value",
            "bh_q_value",
            "stable_examples_mean",
            "variable_examples_mean",
            "stable_examples",
            "variable_examples",
            "any_flip_auc",
        ]
    ].copy()
    uncertainty_table["measure"] = uncertainty_table["measure"].map(
        {
            "predictive_entropy": "Predictive entropy",
            "probability_noise": "Reference-probability SD",
        }
    )
    uncertainty_table = uncertainty_table.rename(
        columns={
            "jev_mode": "Jev mode",
            "measure": "Jev uncertainty measure",
            "examples": "Examples",
            "spearman_rho": "Spearman rho",
            "ci_low": "95% CI low",
            "ci_high": "95% CI high",
            "task_stratified_permutation_p_value": "Within-task permutation p",
            "bh_q_value": "BH q",
            "stable_examples_mean": "Mean when LLMs stable",
            "variable_examples_mean": "Mean when any LLM flips",
            "stable_examples": "Stable examples",
            "variable_examples": "Examples with a flip",
            "any_flip_auc": "Any-flip AUC",
        }
    )
    uncertainty_task_table = jev_uncertainty_by_task.rename(
        columns={
            "jev_mode": "Jev mode",
            "evaluator": "Task",
            "examples": "Examples",
            "spearman_rho": "Spearman rho",
            "p_value": "p",
            "bh_q_value": "BH q",
        }
    ).sort_values(["Jev mode", "Spearman rho"], ascending=[True, False])
    section_introductions = {
        "Model and task differences": (
            "These views share scales across judges and tasks. The relative-accuracy heatmap "
            "subtracts each task's mean, so model-specific strengths are visible even when all "
            "judges score near the top of the raw accuracy scale. Macro F1 gives each reference "
            "label equal weight. N/A marks a task with only one reference class."
        ),
        "Jev request shapes": (
            "The effect plot pairs the two Jev modes on the same examples. Intervals resample "
            "examples, and the p-values use exact McNemar tests on majority correctness."
        ),
        "Jev probability alignment": (
            "For each example, conventional-LLM entropy is the mean label entropy across the "
            "five LLM judges, calculated separately over each judge's ten repetitions. Jev "
            "predictive entropy measures uncertainty within each binary probability estimate. "
            "Probability noise is the standard deviation, across repetitions, of the probability "
            "assigned to the example's reference label."
        ),
        "Variance and uncertainty": (
            "Entropy measures run-to-run label instability across ten repetitions. For these "
            "binary evaluators, zero bits is perfectly stable and one bit is an even split."
        ),
    }
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>Judge sweep {html.escape(sweep_run_id)}</title>",
        (
            "<style>"
            "body{font-family:system-ui,sans-serif;max-width:1600px;margin:2rem auto;"
            "padding:0 1rem}"
            "p,li{line-height:1.5}"
            "table{border-collapse:collapse;font-size:.82rem;width:100%}"
            "th,td{border-bottom:1px solid #ddd;padding:.45rem;text-align:right}"
            "th:first-child,td:first-child{text-align:left}"
            ".plot{margin:2rem 0 4rem}"
            ".findings{max-width:80rem}"
            "details{margin:2rem 0}"
            "</style>"
        ),
        "</head><body>",
        f"<h1>Judge sweep {html.escape(sweep_run_id)}</h1>",
        (
            f"<p>{len(runs):,} base-example runs across "
            f"{runs['judge'].nunique()} judges and {runs['evaluator'].nunique()} evaluators. "
            "Failures remain in every denominator.</p>"
        ),
        (
            f"<p><strong>Exclusions:</strong> {len(excluded_examples)} examples, representing "
            f"{excluded_run_count:,} judge-repetition runs, are omitted under the configured "
            "benchmark exclusions. See <code>excluded-examples.csv</code> for the audit list.</p>"
        ),
        (
            "<p><strong>Human adjudication:</strong> Twenty disputed examples were reviewed on "
            "2026-09-23. No ground-truth labels were changed. Ten examples retained their "
            "existing labels; ten were excluded. One formerly excluded tool-invocation example "
            "was restored with its existing <code>correct</code> label.</p>"
        ),
        "<h2>Class balance</h2>",
        (
            "<p>Precision, recall, and F1 require at least two reference classes in a task. "
            "Single-class tasks retain accuracy, but their classification metrics are reported "
            "as N/A rather than zero.</p>"
        ),
        class_balance_table.to_html(index=False),
        "<section class='findings'><h2>What stands out</h2><ul>",
        *(f"<li>{html.escape(finding)}</li>" for finding in findings),
        "</ul></section>",
    ]
    include_plotly: str | bool = "inline"
    current_section: str | None = None
    for section, title, figure in figures:
        if section != current_section:
            current_section = section
            parts.extend(
                [
                    f"<h2>{html.escape(section)}</h2>",
                    f"<p>{html.escape(section_introductions[section])}</p>",
                ]
            )
        parts.extend(
            [
                f"<section class='plot'><h3>{html.escape(title)}</h3>",
                pio.to_html(
                    figure,
                    full_html=False,
                    include_plotlyjs=include_plotly,
                ),
                "</section>",
            ]
        )
        include_plotly = False
    parts.extend(
        [
            "<h2>Statistical tests</h2>",
            (
                "<p>Cochran's Q tests whether matched majority-correct outcomes differ across "
                "the seven judges for each task. Exact McNemar tests compare pairs on the same "
                "examples. Benjamini-Hochberg q-values control false discovery across the ten "
                "task-level tests and across all pairwise task comparisons, respectively. "
                "Accuracy intervals resample examples, which are the independent sampling unit.</p>"
            ),
            "<h3>Task divergence and omnibus tests</h3>",
            task_table.to_html(index=False, float_format=lambda value: f"{value:.6g}"),
            "<h3>Jev paired comparison</h3>",
            jev_table.to_html(index=False, float_format=lambda value: f"{value:.6g}"),
            "<h3>Jev uncertainty and LLM response entropy</h3>",
            (
                "<p>Spearman intervals resample examples. Permutation tests shuffle Jev values "
                "within each eval task, so task-level differences cannot create the association. "
                "AUC measures how well the Jev quantity separates examples with any conventional "
                "LLM label flip from fully stable examples. This is association, not evidence that "
                "either uncertainty measure causes the other. LLM repeatability also depends on "
                "sampling settings and prompt sensitivity. It is a useful confidence proxy here, "
                "not a direct measurement of internal model confidence.</p>"
            ),
            uncertainty_table.to_html(index=False, float_format=lambda value: f"{value:.6g}"),
            "<details><summary>Jev predictive-entropy alignment by task</summary>",
            uncertainty_task_table.to_html(index=False, float_format=lambda value: f"{value:.6g}"),
            "</details>",
            "<h3>Significant pairwise task comparisons</h3>",
            (
                significant_pair_table.to_html(
                    index=False, float_format=lambda value: f"{value:.6g}"
                )
                if not significant_pair_table.empty
                else "<p>No pairwise task comparisons remain significant at q &lt; 0.05.</p>"
            ),
            "<details><summary>Full metric table</summary>",
            summary.to_html(index=False, na_rep="N/A", float_format=lambda value: f"{value:.6g}"),
            "</details>",
            "<details><summary>Per-class precision, recall, and F1</summary>",
            class_metrics_table.to_html(index=False),
            "</details>",
            "</body></html>",
        ]
    )
    output_path.write_text("".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_url = get_base_url(args.phoenix_endpoint)
    output_directory = args.results_directory / re.sub(r"[^a-zA-Z0-9._-]", "-", args.sweep_run_id)
    output_directory.mkdir(parents=True, exist_ok=True)
    figures_directory = output_directory / "figures"

    loaded_runs = (
        pd.read_csv(args.runs_csv, low_memory=False)
        if args.runs_csv is not None
        else load_runs(sweep_run_id=args.sweep_run_id, base_url=base_url)
    )
    selected_runs = apply_exclusions(select_best_experiments(loaded_runs))
    excluded_runs = selected_runs[selected_runs["exclusion_reason"].notna()].copy()
    runs = selected_runs[selected_runs["exclusion_reason"].isna()].copy()
    excluded_examples = summarize_exclusions(excluded_runs)
    summary = summarize(runs)
    class_balance = summarize_class_balance(runs)
    class_metrics = summarize_class_metrics(runs)
    example_results = get_example_results(runs)
    jev_comparison = compare_jev_modes(example_results)
    (
        jev_uncertainty_alignment,
        jev_uncertainty_tests,
        jev_uncertainty_by_task,
    ) = analyze_jev_uncertainty_alignment(runs)
    task_divergence = analyze_task_divergence(
        example_results=example_results,
        summary=summary,
    )
    pairwise_significance = compare_all_judges(example_results)
    selected_runs.to_csv(output_directory / "runs.csv", index=False)
    excluded_examples.to_csv(output_directory / "excluded-examples.csv", index=False)
    summary.to_csv(output_directory / "summary.csv", index=False)
    class_balance.to_csv(output_directory / "class-balance.csv", index=False)
    class_metrics.to_csv(output_directory / "class-metrics.csv", index=False)
    example_results.to_csv(output_directory / "example-results.csv", index=False)
    jev_comparison.to_csv(output_directory / "jev-comparison.csv", index=False)
    jev_uncertainty_alignment.to_csv(
        output_directory / "jev-uncertainty-alignment.csv", index=False
    )
    jev_uncertainty_tests.to_csv(output_directory / "jev-uncertainty-tests.csv", index=False)
    jev_uncertainty_by_task.to_csv(output_directory / "jev-uncertainty-by-task.csv", index=False)
    task_divergence.to_csv(output_directory / "task-divergence.csv", index=False)
    pairwise_significance.to_csv(output_directory / "pairwise-significance.csv", index=False)
    figures = create_figures(
        runs=runs,
        summary=summary,
        example_results=example_results,
        jev_comparison=jev_comparison,
        jev_uncertainty_alignment=jev_uncertainty_alignment,
        jev_uncertainty_tests=jev_uncertainty_tests,
        jev_uncertainty_by_task=jev_uncertainty_by_task,
        task_divergence=task_divergence,
        pairwise_significance=pairwise_significance,
        figures_directory=figures_directory,
    )
    write_report(
        sweep_run_id=args.sweep_run_id,
        runs=runs,
        summary=summary,
        class_balance=class_balance,
        class_metrics=class_metrics,
        jev_comparison=jev_comparison,
        jev_uncertainty_tests=jev_uncertainty_tests,
        jev_uncertainty_by_task=jev_uncertainty_by_task,
        task_divergence=task_divergence,
        pairwise_significance=pairwise_significance,
        excluded_examples=excluded_examples,
        excluded_run_count=len(excluded_runs),
        figures=figures,
        output_path=output_directory / "report.html",
    )
    print(output_directory.resolve())


if __name__ == "__main__":
    main()
