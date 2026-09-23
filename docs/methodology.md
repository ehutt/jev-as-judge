# Methodology

## Research question

The benchmark asks how two Jev request shapes compare with conventional language-model judges on fixed classification tasks. It measures three distinct properties:

1. Accuracy against authored or dataset-provided labels.
2. Repeatability when the input does not change.
3. Operational cost and latency under the same runner.

These results apply to the examples, prompts, models, and provider behavior recorded by a run. They do not establish a universal judge ranking.

## Experimental unit

The benchmark has 532 base examples across ten tasks. The default configuration runs each example ten times per judge. One full sweep therefore has 5,320 model calls per judge and 37,240 calls across seven judges.

The base example is the independent sampling unit. Repetitions measure judge variability and are not treated as additional independent examples in confidence intervals.

## Inputs and labels

Nine suites use checked-in examples written to probe the decision boundary in the matching Phoenix evaluator rubric. Correctness expands each question into one correct and one incorrect response. The other authored suites store their expected label next to the input.

The PII suite uses a fixed, checked-in 150-record sample from NVIDIA Nemotron-PII. Every sampled record contains at least one annotated PII category. The suite therefore measures detection rate, which is recall on this all-positive slice. It cannot measure precision or false-positive rate.

## Prompt conditions

The conventional language-model judges use Phoenix's built-in classification evaluator factories. The benchmark supplies the same input record and uses the provider defaults exposed by the AI SDK adapters.

`jev-slot-in` renders the complete Phoenix template with the input record, flattens the messages to text, and sends that text as Jev state. The Jev question asks for one of the original Phoenix labels.

`jev-native` sends only the example data as state. It expresses the evaluation instruction and each label criterion in Jev's typed question structure. Both Jev modes use the same response mapping, validation, tracing, and metric code.

The copied templates come from Phoenix commit `6f03f903b8d3eddffe11e6b695c1c24e244f946a`. Offline tests check their SHA-256 digests so prompt changes require an explicit provenance update.

## Parallelism and rate limits

The runner creates one lane for each provider. Lanes execute in parallel. Judges within a provider lane execute in sequence so they do not compete with each other for the same provider limit.

Vitest runs the ten suite files in parallel and runs cases within each file concurrently. The default per-suite limit is 1 for Jev and 10 for the language-model judges. `SWEEP_CONCURRENCY` overrides that value for every selected judge.

The AI SDK retries a transient provider failure twice. A call that still fails remains in the exported run set and counts as incorrect.

## Metrics

Accuracy is the mean of the per-run correctness indicator. The 95 percent interval bootstraps base examples and gives each example its mean over repetitions before resampling.

Macro precision, recall, and F1 operate on the expected and predicted labels. Cohen's kappa measures agreement with the reference labels after accounting for label prevalence.

Consistency is the share of repeated labels matching an example's modal label. Flip rate is the share of examples with more than one observed label. Label entropy measures the same variation in bits.

Pairwise judge comparisons use exact McNemar tests on matched majority-correct outcomes. Benjamini-Hochberg correction controls the false-discovery rate across the comparison family. Cochran's Q tests the matched outcomes across all seven judges for each task.

Jev's top-label Brier score uses the probability attached to the selected label. The uncertainty analysis compares Jev predictive entropy and probability variation with label entropy from the conventional judges. Those comparisons are associations, not evidence that one uncertainty measure causes another.

Latency is elapsed evaluator time recorded by Phoenix. Cost uses Phoenix's trace-linked token totals and the price snapshot stored in experiment metadata. The analysis recalculates cost from that snapshot and reports disagreements with Phoenix's model-price table.

## Exclusions and duplicate experiments

The analyzer selects one experiment for each judge and task when a resumed run created duplicates. It prefers the experiment with the most rows, then the fewest errors, then the most recent experiment ID. The raw selection remains visible in `runs.csv` through experiment IDs.

Nine tool-invocation examples are temporarily excluded from comparative statistics because their labels depend on disputed assumptions about whether the user supplied enough information for a tool call. `excluded-examples.csv` records the exact examples and reason. The underlying Phoenix runs are not deleted.

## Interpretation limits

- The authored tasks are useful for rubric boundary testing but are not a representative sample of production traffic.
- One fixed PII slice cannot measure false positives.
- Model aliases and hosted inference systems can change after publication.
- Provider defaults can include undocumented or changing sampling behavior.
- Cost comparisons depend on the prompt lengths in this suite and the price snapshot date.
- Low variance is useful only when the repeated decision is aligned with the reference label.
