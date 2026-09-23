# Jev as a judge

This repository collects experiments and benchmarks for evaluating Jev as a judge.

## Benchmarks

| Benchmark | What it tests |
| --- | --- |
| [Phoenix Evals](benchmarks/phoenix-evals/README.md) | Two Jev request formats and five language-model judges across ten Phoenix classification tasks |

Each benchmark keeps its source, fixtures, dependency locks, analysis scripts, and reproduction instructions in its own directory.

## Add a benchmark

Create a directory under `benchmarks/` with a README that explains the research question, setup, execution, analysis, and reproducibility limits. Keep generated datasets, model outputs, reports, and charts out of version control unless they are required source fixtures with clear provenance and licensing.

This repository uses the [Elastic License 2.0](LICENSE).
