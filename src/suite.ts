import { getJudge, getJudgeId, PRICING_DATE } from "./judges.js";

export const PHOENIX_EVALS_SOURCE_SHA =
  "6f03f903b8d3eddffe11e6b695c1c24e244f946a";
export const DEFAULT_REPETITIONS = 10;

function getRepetitions(): number {
  const value = process.env.SWEEP_REPETITIONS;
  if (value === undefined) {
    return DEFAULT_REPETITIONS;
  }
  const repetitions = Number.parseInt(value, 10);
  if (!Number.isInteger(repetitions) || repetitions < 1) {
    throw new Error(
      `SWEEP_REPETITIONS must be a positive integer, received ${JSON.stringify(value)}`
    );
  }
  return repetitions;
}

function getSweepRunId(): string {
  return (
    process.env.SWEEP_RUN_ID ??
    `manual-${new Date().toISOString().slice(0, 10)}`
  );
}

export function getSuiteConfig(evaluator: string) {
  const judgeId = getJudgeId();
  const judge = getJudge(judgeId);
  const repetitions = getRepetitions();
  return {
    datasetName: `judge-sweep/${evaluator}`,
    repetitions,
    metadata: {
      judge: judgeId,
      provider: judge.provider,
      variant: judge.variant,
      modelId: judge.modelId,
      pricing: {
        date: PRICING_DATE,
        inputPerMillionTokensUsd: judge.pricing.input,
        outputPerMillionTokensUsd: judge.pricing.output,
        source: judge.pricingSource,
      },
      repetitions,
      phoenixEvalsSha: PHOENIX_EVALS_SOURCE_SHA,
      sweepRunId: getSweepRunId(),
      date: new Date().toISOString().slice(0, 10),
    },
  };
}
