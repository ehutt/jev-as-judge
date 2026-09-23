import type { EvaluatorName } from "../templates/index.js";
import { completenessNativeSpec } from "./completeness.js";
import { concisenessNativeSpec } from "./conciseness.js";
import { correctnessNativeSpec } from "./correctness.js";
import { hallucinationNativeSpec } from "./hallucination.js";
import { piiDetectionNativeSpec } from "./pii_detection.js";
import { refusalNativeSpec } from "./refusal.js";
import { retrievalRelevanceNativeSpec } from "./retrieval_relevance.js";
import { toolInvocationNativeSpec } from "./tool_invocation.js";
import { toolResponseHandlingNativeSpec } from "./tool_response_handling.js";
import type { NativeQuestionSpec } from "./types.js";
import { userFrictionNativeSpec } from "./user_friction.js";

export const NATIVE_SPECS: Record<EvaluatorName, NativeQuestionSpec> = {
  correctness: correctnessNativeSpec,
  conciseness: concisenessNativeSpec,
  hallucination: hallucinationNativeSpec,
  refusal: refusalNativeSpec,
  retrieval_relevance: retrievalRelevanceNativeSpec,
  tool_invocation: toolInvocationNativeSpec,
  tool_response_handling: toolResponseHandlingNativeSpec,
  completeness: completenessNativeSpec,
  user_friction: userFrictionNativeSpec,
  pii_detection: piiDetectionNativeSpec,
};

export { buildNativeRequest } from "./types.js";
