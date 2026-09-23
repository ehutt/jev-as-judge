import { EVALUATOR_CONFIGS } from "../templates/index.js";
import type { NativeQuestionSpec } from "./types.js";

export const toolInvocationNativeSpec = {
  config: EVALUATOR_CONFIGS.tool_invocation,
  buildState: (record) => ({
    input: String(record.input),
    availableTools: String(record.availableTools),
    toolSelection: String(record.toolSelection),
  }),
  criteria: {
    correct:
      "Every invocation is well formed, schema-valid, complete, faithful to the conversation, and safe.",
    incorrect:
      "At least one invocation is malformed, missing or inventing fields, has wrong values, or exposes sensitive data.",
  },
} satisfies NativeQuestionSpec;
