import { EVALUATOR_CONFIGS } from "../templates/index.js";
import type { NativeQuestionSpec } from "./types.js";

export const correctnessNativeSpec = {
  config: EVALUATOR_CONFIGS.correctness,
  buildState: (record) => ({
    input: String(record.input),
    output: String(record.output),
  }),
  criteria: {
    correct:
      "Accurate, complete, logically consistent, precise, and not misleading.",
    incorrect:
      "Contains a factual, completeness, terminology, ambiguity, or consistency failure.",
  },
} satisfies NativeQuestionSpec;
