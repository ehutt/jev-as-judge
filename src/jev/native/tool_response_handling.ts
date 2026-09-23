import { EVALUATOR_CONFIGS } from "../templates/index.js";
import type { NativeQuestionSpec } from "./types.js";

export const toolResponseHandlingNativeSpec = {
  config: EVALUATOR_CONFIGS.tool_response_handling,
  buildState: (record) => ({
    input: String(record.input),
    toolCall: String(record.toolCall),
    toolResult: String(record.toolResult),
    output: String(record.output),
  }),
  criteria: {
    correct:
      "Accurately uses every tool result, handles errors appropriately, and protects sensitive information.",
    incorrect:
      "Hallucinates, misreads, ignores, transforms incorrectly, retries badly, or leaks tool-result data.",
  },
} satisfies NativeQuestionSpec;
