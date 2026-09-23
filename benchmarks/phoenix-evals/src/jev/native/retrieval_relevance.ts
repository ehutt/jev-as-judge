import { EVALUATOR_CONFIGS } from "../templates/index.js";
import type { NativeQuestionSpec } from "./types.js";

export const retrievalRelevanceNativeSpec = {
  config: EVALUATOR_CONFIGS.retrieval_relevance,
  buildState: (record) => ({
    input: String(record.input),
    context: String(record.context),
  }),
  criteria: {
    relevant:
      "Contains material that helps address the request, including partial or outdated material about the right subject.",
    irrelevant:
      "Provides no useful material for the request, including empty, failed, tangential, or wrong-subject retrievals.",
  },
} satisfies NativeQuestionSpec;
