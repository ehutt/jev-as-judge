import { EVALUATOR_CONFIGS } from "../templates/index.js";
import type { NativeQuestionSpec } from "./types.js";

export const userFrictionNativeSpec = {
  config: EVALUATOR_CONFIGS.user_friction,
  buildState: (record) => ({
    conversation: String(record.conversation),
    userMessage: String(record.userMessage),
  }),
  criteria: {
    friction:
      "The latest message corrects, retries, expresses frustration with, or challenges the assistant's preceding behavior.",
    no_friction:
      "The latest message is an ordinary answer, refinement, new task, rejection, question, or ambiguous response without expressed friction.",
  },
} satisfies NativeQuestionSpec;
