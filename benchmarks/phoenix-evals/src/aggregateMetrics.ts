/**
 * Suite-level confusion-matrix metrics for the eval benchmarks.
 *
 * The per-case `accuracy` gate is blind to class imbalance — a judge that
 * always predicts the majority label can pass it. Each suite therefore also
 * accumulates its ground-truth and predicted labels and registers a trailing
 * test that scores them with macro precision / recall / F1 (annotation names
 * `precision`, `recall`, `f1`), which the suite gates via `acceptanceCriteria`.
 *
 * Tracked upstream: native confusion-matrix acceptance criteria in the
 * phoenix-client harness would remove the synthetic trailing test —
 * https://github.com/Arize-ai/phoenix/issues/14020
 */
import * as px from "@arizeai/phoenix-client/vitest";
import { createPrecisionRecallFScoreEvaluators } from "@arizeai/phoenix-evals";
import { afterEach, beforeEach } from "vitest";

const TRAILING_TEST_TIMEOUT_MS = 600_000;
const COMPLETION_POLL_INTERVAL_MS = 100;

export interface LabelAccumulator {
  /** Ground-truth labels, one per completed case. */
  truth: string[];
  /** Evaluator-under-test predictions, aligned by index with `truth`. */
  predicted: string[];
}

/** Creates the accumulator a suite threads through its cases. */
export function createLabelAccumulator(): LabelAccumulator {
  return { truth: [], predicted: [] };
}

/**
 * Records one case's ground-truth vs predicted label pair.
 * Skips the case when either side is missing so the sequences stay aligned.
 *
 * @param params - the recording parameters
 * @param params.labels - the suite's accumulator
 * @param params.truth - the ground-truth label from the example's `expected`
 * @param params.predicted - the label the evaluator-under-test produced
 */
export function recordPrediction({
  labels,
  truth,
  predicted,
}: {
  labels: LabelAccumulator;
  truth: string | undefined;
  predicted: string | undefined;
}): void {
  if (typeof truth !== "string" || typeof predicted !== "string") {
    return;
  }
  labels.truth.push(truth);
  labels.predicted.push(predicted);
}

const completion = { started: 0, finished: 0, waiting: 0, tracked: false };

/**
 * The sweep runs a file's tests concurrently (`sequence.concurrent`), so a
 * trailing test can start while earlier cases are still in flight. Vitest
 * starts tests in declaration order, so once a trailing test has started,
 * every earlier test has started too; the cases are done once every started
 * test has finished except the trailing tests themselves, which are waiting.
 *
 * Call inside `px.describe`, then await the returned function at the top of
 * each trailing test. Counters are per test file (vitest isolates modules).
 */
export function registerCompletionTracker(): () => Promise<void> {
  if (!completion.tracked) {
    completion.tracked = true;
    beforeEach(() => {
      completion.started += 1;
    });
    afterEach(() => {
      completion.finished += 1;
    });
  }
  return async function waitForEarlierTests() {
    completion.waiting += 1;
    try {
      while (completion.finished < completion.started - completion.waiting) {
        await new Promise((resolve) =>
          setTimeout(resolve, COMPLETION_POLL_INTERVAL_MS)
        );
      }
    } finally {
      completion.waiting -= 1;
    }
  };
}

/**
 * Registers the trailing test that scores the accumulated labels with macro
 * precision / recall / F1. Call it inside `px.describe` after every
 * `px.test.each` so it is declared — and therefore starts — last. It runs
 * once regardless of the suite's `repetitions` and waits for every case to
 * finish before scoring.
 *
 * @param labels - the suite's accumulator, filled by the preceding cases
 */
export function registerAggregateMetricsTest(labels: LabelAccumulator): void {
  const { precision, recall, fScore } = createPrecisionRecallFScoreEvaluators({
    average: "macro",
  });
  const waitForEarlierTests = registerCompletionTracker();
  px.test(
    "aggregate metrics: macro precision/recall/F1 across all cases",
    {
      input: {
        description:
          "Macro precision/recall/F1 of the evaluator-under-test's labels against the ground truth across every case in this suite",
      },
      repetitions: 1,
    },
    async () => {
      await waitForEarlierTests();
      const batch = { expected: labels.truth, output: labels.predicted };
      px.logOutput(batch);
      await px.evaluate(precision, batch);
      await px.evaluate(recall, batch);
      await px.evaluate(fScore, batch);
    },
    TRAILING_TEST_TIMEOUT_MS
  );
}
