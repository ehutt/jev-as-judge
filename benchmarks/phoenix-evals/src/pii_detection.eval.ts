/**
 * PII-detection evaluator benchmark
 *
 * Reads the checked-in JSONL at `fixtures/pii_detection.nemotron.jsonl`, a
 * 150-record sample of the public nvidia/Nemotron-PII dataset. Fourteen records
 * with unresolved source-taxonomy conflicts are listed in
 * `fixtures/pii_detection.exclusions.json`, leaving 136 scored examples. The
 * fixed files keep the suite deterministic and offline.
 *
 * Nemotron-PII is a span-annotated NER corpus that is ~99% positive, so this
 * suite measures only the binary DETECTION RATE: given realistic PII-bearing
 * text, does the evaluator score `pii_detected`? Because there are effectively
 * no negatives, precision / false-positive rate cannot be measured here. For a
 * balanced precision/recall suite, see `pii_detection.synthetic.eval.ts`.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import * as px from "@arizeai/phoenix-client/vitest";

import { registerCompletionTracker } from "./aggregateMetrics.js";
import { accuracy } from "./evaluators.js";
import { resolveEvaluator } from "./resolve.js";
import { getSuiteConfig } from "./suite.js";

type PiiLabel = "pii_detected" | "no_pii_detected";

type NemotronRecord = {
  uid: string;
  domain: string;
  document_type: string;
  document_format: string;
  locale: string;
  text: string;
  pii_categories: string[];
  expected_label: PiiLabel;
};

type PiiExclusion = {
  uid: string;
  reason: string;
};

const fixturePath = fileURLToPath(
  new URL("./fixtures/pii_detection.nemotron.jsonl", import.meta.url)
);
const allRecords = readFileSync(fixturePath, "utf-8")
  .trim()
  .split("\n")
  .map((line) => JSON.parse(line) as NemotronRecord);
const exclusionsPath = fileURLToPath(
  new URL("./fixtures/pii_detection.exclusions.json", import.meta.url)
);
const exclusions = JSON.parse(
  readFileSync(exclusionsPath, "utf-8")
) as PiiExclusion[];
const excludedUids = new Set(exclusions.map(({ uid }) => uid));
const records = allRecords.filter(({ uid }) => !excludedUids.has(uid));

if (allRecords.length - records.length !== exclusions.length) {
  throw new Error("PII exclusion list contains a UID that is missing or duplicated");
}

const evaluator = resolveEvaluator("pii_detection");

const cases = records.map((record) => ({
  input: { conversation: record.text },
  expected: { label: record.expected_label },
  metadata: {
    uid: record.uid,
    domain: record.domain,
    document_type: record.document_type,
    document_format: record.document_format,
    locale: record.locale,
    pii_categories: record.pii_categories,
  },
  splits: [record.document_format, record.locale],
}));

// Detection-rate accumulator: on an all-positive dataset, accuracy equals the
// fraction of PII-bearing documents the evaluator correctly flags.
let detected = 0;
let scored = 0;

px.describe(
  "pii-detection-benchmark",
  () => {
    px.test.each(cases)(
      (row) =>
        `[${String(row.metadata?.document_format)}/${String(
          row.metadata?.locale
        )}] ${String(row.metadata?.domain)} (${String(row.metadata?.uid)})`,
      async ({ input, expected }) => {
        const result = await evaluator.evaluate(input);
        px.logOutput(result);
        px.logAnnotation({
          name: "pii_detection",
          label: result.label,
          explanation: result.explanation,
          annotatorKind: "LLM",
        });
        scored += 1;
        if (result.label === expected?.label) {
          detected += 1;
        }
        await px.evaluate(accuracy);
      }
    );

    const waitForEarlierTests = registerCompletionTracker();
    px.test(
      "detection rate: fraction of PII-bearing documents flagged pii_detected",
      {
        input: {
          description: "Detection rate across every case in this suite",
        },
        repetitions: 1,
      },
      async () => {
        await waitForEarlierTests();
        const detectionRate = scored === 0 ? 0 : detected / scored;
        px.logOutput({ detected, scored, detectionRate });
      },
      600_000
    );
  },
  {
    description:
      "PII detection rate on 136 records from a stratified nvidia/Nemotron-PII sample (structured/unstructured x US/intl), after excluding 14 source-taxonomy conflicts. Every retained case contains at least one source-annotated PII category, so accuracy is reported as positive-slice detection rate.",
    ...getSuiteConfig("pii_detection"),
    acceptanceCriteria: [
      { annotationName: "accuracy", metric: "average", threshold: 0.9 },
    ],
  }
);
