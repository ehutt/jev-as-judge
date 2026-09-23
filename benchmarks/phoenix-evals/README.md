# Phoenix Evals benchmark

This benchmark compares two ways of using Jev as an evaluator with five language-model judges. Every judge sees the same 532 examples across ten Phoenix classification tasks. The default experiment repeats each example ten times and records accuracy, consistency, latency, token usage, and cost in Phoenix.

This is a research benchmark, not a general ranking of model quality. Most cases are purpose-built boundary examples. The PII task uses a fixed sample of NVIDIA's synthetic Nemotron-PII dataset.

## What the benchmark runs

| Task | Base examples | Ground-truth labels |
| --- | ---: | --- |
| Correctness | 80 | correct / incorrect |
| Conciseness | 34 | concise / verbose |
| Hallucination | 37 | grounded / hallucinated |
| Refusal | 40 | refusal / non-refusal |
| Retrieval relevance | 33 | relevant / irrelevant |
| Tool invocation | 22 | correct / incorrect |
| Tool-response handling | 51 | correct / incorrect |
| Completeness | 45 | complete / incomplete |
| User friction | 40 | friction / no friction |
| PII detection | 150 | PII detected |

The two Jev conditions differ only in request shape:

- `jev-slot-in` renders the original Phoenix evaluator prompt and places the complete text in Jev's `state` field.
- `jev-native` puts example data in `state` and moves the rubric into Jev's typed `instructions` and `criteria` fields.

The comparison judges use the built-in Phoenix evaluators without benchmark-specific prompt tuning.

| Judge ID | Requested model | Price snapshot, input/output per million tokens |
| --- | --- | ---: |
| `jev-slot-in` | `jev-latest` | $0.042 / $0 |
| `jev-native` | `jev-latest` | $0.042 / $0 |
| `openai-cheap` | `gpt-5-nano` | $0.05 / $0.40 |
| `openai-frontier` | `gpt-5.6-sol` | $4 / $20 |
| `anthropic-cheap` | `claude-haiku-4-5` | $1 / $5 |
| `anthropic-frontier` | `claude-opus-5` | $5 / $25 |
| `gemini` | `gemini-3.8-flash` | $0.75 / $3.75 |

Prices are the rates recorded on 2026-09-21. Provider defaults are intentional. The benchmark does not override temperature, reasoning effort, or thinking budgets.

## Requirements

- Node.js 24.20.0
- pnpm 12.0.0 through Corepack
- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- Phoenix 20.15.0 or a compatible hosted Phoenix instance
- API keys for the judges you select

The Node and Python dependency graphs are locked. Hosted model aliases can still change upstream. Each experiment records the requested model, the resolved model returned by the provider, the run date, the pricing snapshot, and the source commit for the Phoenix evaluator templates.

## Set up the project

Clone the repository and install the locked dependencies:

```bash
git clone https://github.com/AparnaDhinakaran/jev-as-judge.git
cd jev-as-judge/benchmarks/phoenix-evals
corepack enable
pnpm install --frozen-lockfile
cp .env.example .env
```

Add the keys required by the judges you plan to run:

| Provider | Environment variable |
| --- | --- |
| TypeSafe | `TYPESAFE_AI_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google | `GOOGLE_GENERATIVE_AI_API_KEY` |

Start Phoenix in a separate terminal. This command runs the server version used to build the benchmark:

```bash
uvx --from arize-phoenix==20.15.0 phoenix serve
```

By default, the benchmark connects to `http://localhost:6006`. Set `PHOENIX_ENDPOINT` and `PHOENIX_API_KEY` in `.env` for another instance. Leave `PHOENIX_WORKING_DIR` unset unless you explicitly want a separate Phoenix database.

Check the local setup without calling a model:

```bash
pnpm verify
```

This type-checks the TypeScript, runs the offline unit tests, and verifies that the locked Python analysis environment can start.

## Run a smoke test first

A full run makes 37,240 paid evaluator calls. Start with one repetition, one task, and one judge:

```bash
SWEEP_RUN_ID=smoke-$(date -u +%Y%m%dT%H%M%SZ) \
SWEEP_REPETITIONS=1 \
JUDGE=jev-native \
pnpm evals -- src/correctness.eval.ts
```

Repeat with a comparison judge:

```bash
SWEEP_RUN_ID=smoke-$(date -u +%Y%m%dT%H%M%SZ) \
SWEEP_REPETITIONS=1 \
JUDGE=openai-cheap \
pnpm evals -- src/correctness.eval.ts
```

Inspect both experiments in Phoenix. Each evaluator run should contain an `EVALUATOR` span with one `LLM` child span, token counts, the resolved model, and cost.

The runner can also execute the same smoke comparison across all ten tasks:

```bash
SWEEP_REPETITIONS=1 pnpm sweep -- jev-native openai-cheap
```

## Reproduce the full experiment

Run all seven judges:

```bash
pnpm sweep
```

The runner creates one lane per provider. Provider lanes run in parallel, while judges that share a provider run in sequence. Each evaluator suite runs its cases concurrently. Set `SWEEP_CONCURRENCY` to lower the per-suite limit if a provider returns rate-limit errors.

Every run is assigned a `SWEEP_RUN_ID`. Results, complete Vitest logs, and Phoenix reporter artifacts go to `analysis/results/<sweep-run-id>/`. The directory is ignored by Git.

Run or resume a subset with an explicit ID:

```bash
SWEEP_RUN_ID=20260922-reproduction-1 \
pnpm sweep -- jev-slot-in jev-native openai-cheap
```

The runner continues after a judge exits non-zero because an acceptance-gate failure is benchmark data. It writes failed judge IDs to `failed-judges.txt`. Review the log before resuming that judge with the same run ID.

## Analyze a run

The analysis reads the experiments for one `SWEEP_RUN_ID` from Phoenix and joins the GraphQL cost summary:

```bash
pnpm analyze 20260922-reproduction-1
```

The command writes these artifacts under `analysis/results/<sweep-run-id>/`:

- `runs.csv` with one row per judge, task, example, and repetition, including excluded rows and their reasons
- `summary.csv` with accuracy, bootstrap intervals, macro metrics, agreement, consistency, latency, token usage, cost, failures, and Jev Brier score
- paired comparison and task-divergence CSV files
- Jev uncertainty and conventional-judge variability CSV files
- `report.html` with self-contained interactive charts
- PNG figures for publication or review

After the first export, regenerate every derived table and chart without Phoenix by passing the saved run-level CSV:

```bash
pnpm analyze 20260922-reproduction-1 \
  --runs-csv analysis/results/20260922-reproduction-1/runs.csv
```

Failures stay in the accuracy denominator. Accuracy intervals resample base examples, which are the independent sampling unit. Judge variation across repeated calls is reported separately as consistency, flip rate, and label entropy. See [docs/methodology.md](docs/methodology.md) for definitions and analysis decisions.

## Reproducibility boundaries

The benchmark fixes the source examples, copied evaluator templates, requested model IDs, package versions, analysis code, and statistical seeds. It does not freeze hosted model weights or provider infrastructure. Re-running an alias such as `jev-latest` at a later date may not call the same model version. Network latency and provider-side retries also vary.

For a publishable reproduction, keep the complete output directory and report:

- the Git commit for this repository
- the `SWEEP_RUN_ID`
- the Phoenix version and endpoint type
- the start date and region
- the requested and resolved model IDs from `runs.csv`
- any resumed judges or excluded examples

Do not compare only successful rows. The analysis treats timeouts, parse failures, and provider errors as incorrect and reports coverage separately.

## Project layout

```text
analysis/
  analyze.py                 Phoenix export, statistics, and report generation
  results/                   Generated outputs, ignored except for .gitkeep
docs/
  methodology.md             Experiment and analysis protocol
src/
  *.eval.ts                  Ten benchmark suites and fixed examples
  fixtures/                  Checked-in Nemotron-PII sample
  jev/                       Jev adapter, native request specs, copied templates
setup-phoenix-costs.ts        Checks or registers the benchmark price snapshot
sweep.ts                     Cross-provider runner
```

Template and dataset attribution is in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The project is distributed under the [Elastic License 2.0](../../LICENSE).
