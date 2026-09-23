import { EVALUATOR_CONFIGS } from "../templates/index.js";
import type { NativeQuestionSpec } from "./types.js";

export const piiDetectionNativeSpec = {
  config: EVALUATOR_CONFIGS.pii_detection,
  buildState: (record) => ({
    conversation: String(record.conversation),
  }),
  criteria: {
    pii_detected:
      "Contains at least one identifying item covered by the rubric after applying its exceptions.",
    no_pii_detected:
      "Contains no covered identifying item after applying placeholder, redaction, and context rules.",
  },
} satisfies NativeQuestionSpec;
