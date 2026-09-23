import { EVALUATOR_CONFIGS } from "../templates/index.js";
import type { NativeQuestionSpec } from "./types.js";

export const completenessNativeSpec = {
  config: EVALUATOR_CONFIGS.completeness,
  buildState: (record) => ({
    conversation: String(record.conversation),
  }),
  criteria: {
    complete:
      "Every active, non-withdrawn request is observably fulfilled, or there is no substantive active request.",
    incomplete:
      "At least one active, non-withdrawn request is pending, blocked, failed, or ignored.",
  },
} satisfies NativeQuestionSpec;
