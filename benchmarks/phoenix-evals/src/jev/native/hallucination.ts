import { EVALUATOR_CONFIGS } from "../templates/index.js";
import type { NativeQuestionSpec } from "./types.js";

export const hallucinationNativeSpec = {
  config: EVALUATOR_CONFIGS.hallucination,
  buildState: (record) => ({
    input: String(record.input),
    output: String(record.output),
  }),
  criteria: {
    hallucinated:
      "The response contains at least one claim unsupported by or contradictory to the conversation.",
    grounded:
      "Every response claim is supported by the conversation or allowed general knowledge.",
  },
} satisfies NativeQuestionSpec;
