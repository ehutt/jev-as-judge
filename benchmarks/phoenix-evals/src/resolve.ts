import {
  createCompletenessEvaluator,
  createConcisenessEvaluator,
  createCorrectnessEvaluator,
  createHallucinationEvaluator,
  createPiiDetectionEvaluator,
  createRefusalEvaluator,
  createRetrievalRelevanceEvaluator,
  createToolInvocationEvaluator,
  createToolResponseHandlingEvaluator,
  createUserFrictionEvaluator,
  formatTemplate,
  type EvaluationResult,
} from "@arizeai/phoenix-evals";
import { type LanguageModelMiddleware, wrapLanguageModel } from "ai";

import {
  createJevEvaluator,
  type JevEvaluationResult,
} from "./jev/createJevEvaluator.js";
import { buildNativeRequest, NATIVE_SPECS } from "./jev/native/index.js";
import { flattenTemplate, type JevQuestionRequest } from "./jev/native/types.js";
import {
  EVALUATOR_CONFIGS,
  type EvaluatorName,
} from "./jev/templates/index.js";
import {
  createLanguageModel,
  type ConcreteLanguageModel,
  getJudge,
  getJudgeId,
  type JudgeId,
} from "./judges.js";

export type BenchmarkEvaluationResult = EvaluationResult & {
  metadata?: Record<string, unknown>;
};

export type BenchmarkEvaluator<
  RecordType extends Record<string, unknown> = Record<string, unknown>,
> = {
  name: string;
  kind: "LLM" | "CODE";
  evaluate: (
    record: RecordType
  ) => Promise<BenchmarkEvaluationResult | JevEvaluationResult>;
};

function createBuiltInEvaluator({
  evaluatorName,
  model,
}: {
  evaluatorName: EvaluatorName;
  model: ConcreteLanguageModel;
}): BenchmarkEvaluator<Record<string, unknown>> {
  switch (evaluatorName) {
    case "correctness":
      return createCorrectnessEvaluator({ model });
    case "conciseness":
      return createConcisenessEvaluator({ model });
    case "hallucination":
      return createHallucinationEvaluator({ model });
    case "refusal":
      return createRefusalEvaluator({ model });
    case "retrieval_relevance":
      return createRetrievalRelevanceEvaluator({ model });
    case "tool_invocation":
      return createToolInvocationEvaluator({ model });
    case "tool_response_handling":
      return createToolResponseHandlingEvaluator({ model });
    case "completeness":
      return createCompletenessEvaluator({ model });
    case "user_friction":
      return createUserFrictionEvaluator({ model });
    case "pii_detection":
      return createPiiDetectionEvaluator({ model });
  }
}

function createTrackedLanguageModel({
  judgeId,
  onResolvedModel,
}: {
  judgeId: JudgeId;
  onResolvedModel: (modelId: string) => void;
}): ConcreteLanguageModel {
  const middleware: LanguageModelMiddleware = {
    specificationVersion: "v4",
    async wrapGenerate({ doGenerate, model }) {
      const result = await doGenerate();
      onResolvedModel(result.response?.modelId ?? model.modelId);
      return result;
    },
  };
  return wrapLanguageModel({
    model: createLanguageModel(judgeId),
    middleware,
  });
}

function createLanguageModelEvaluator<
  RecordType extends Record<string, unknown>,
>({
  evaluatorName,
  judgeId,
}: {
  evaluatorName: EvaluatorName;
  judgeId: JudgeId;
}): BenchmarkEvaluator<RecordType> {
  const judge = getJudge(judgeId);
  return {
    name: evaluatorName,
    kind: "LLM",
    async evaluate(record) {
      let resolvedModel: string = judge.modelId;
      const model = createTrackedLanguageModel({
        judgeId,
        onResolvedModel: (modelId) => {
          resolvedModel = modelId;
        },
      });
      const evaluator = createBuiltInEvaluator({ evaluatorName, model });
      const result = await evaluator.evaluate(record);
      return {
        ...result,
        metadata: {
          model: judge.modelId,
          resolvedModel,
        },
      };
    },
  };
}

export function getSlotInRequest({
  evaluatorName,
  record,
}: {
  evaluatorName: EvaluatorName;
  record: Record<string, unknown>;
}): JevQuestionRequest {
  const config = EVALUATOR_CONFIGS[evaluatorName];
  return {
    state: flattenTemplate(
      formatTemplate({ template: config.template, variables: record })
    ),
    instructions:
      "Select the label that best satisfies the complete rubric in the state.",
    criteria: Object.fromEntries(
      Object.keys(config.choices).map((label) => [label, null])
    ),
  };
}

function createJevBenchmarkEvaluator<
  RecordType extends Record<string, unknown>,
>({
  evaluatorName,
  variant,
}: {
  evaluatorName: EvaluatorName;
  variant: "slot-in" | "native";
}): BenchmarkEvaluator<RecordType> {
  const config = EVALUATOR_CONFIGS[evaluatorName];
  const nativeSpec = NATIVE_SPECS[evaluatorName];
  return createJevEvaluator<RecordType>({
    name: evaluatorName,
    choices: config.choices,
    buildRequest: (record) =>
      variant === "slot-in"
        ? getSlotInRequest({ evaluatorName, record })
        : buildNativeRequest({ spec: nativeSpec, record }),
  });
}

export function resolveEvaluator<
  RecordType extends Record<string, unknown> = Record<string, unknown>,
>(evaluatorName: EvaluatorName): BenchmarkEvaluator<RecordType> {
  const judgeId = getJudgeId();
  const judge = getJudge(judgeId);
  if (judge.variant === "slot-in" || judge.variant === "native") {
    return createJevBenchmarkEvaluator({
      evaluatorName,
      variant: judge.variant,
    });
  }
  return createLanguageModelEvaluator({ evaluatorName, judgeId });
}
