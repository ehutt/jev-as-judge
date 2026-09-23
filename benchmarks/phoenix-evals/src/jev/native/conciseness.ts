import { EVALUATOR_CONFIGS } from "../templates/index.js";
import type { NativeQuestionSpec } from "./types.js";

export const concisenessNativeSpec = {
  config: EVALUATOR_CONFIGS.conciseness,
  buildState: (record) => ({
    input: String(record.input),
    output: String(record.output),
  }),
  criteria: {
    concise:
      "Directly answers the request with the necessary detail and no needless content.",
    verbose:
      "Contains filler, repetition, meta-commentary, unnecessary hedging, or unsolicited detail.",
  },
} satisfies NativeQuestionSpec;
