import { typeSafeAi } from "@ai-sdk/typesafe-ai";
import {
  createEvaluator,
  type ClassificationChoicesMap,
} from "@arizeai/phoenix-evals";
import {
  type Span,
  SpanStatusCode,
  trace,
  type Tracer,
} from "@opentelemetry/api";
import { experimental_evaluate, type Experimental_EvaluationModel } from "ai";

import type { JevQuestionRequest } from "./native/types.js";

const JSON_MIME_TYPE = "application/json";
const DEFAULT_MODEL_ID = "jev-latest";

export type JevEvaluationMetadata = {
  model: string;
  resolvedModel: string;
  probabilities: Record<string, number>;
  confidence: number | undefined;
  usage: {
    inputTokens: number | undefined;
    outputTokens: number | undefined;
    totalTokens: number | undefined;
  };
};

export type JevEvaluationResult = {
  label: string;
  score: number;
  explanation?: undefined;
  metadata: JevEvaluationMetadata;
};

export type JevEvaluator<
  RecordType extends Record<string, unknown> = Record<string, unknown>,
> = {
  name: string;
  kind: "LLM";
  evaluate: (record: RecordType) => Promise<JevEvaluationResult>;
};

export type CreateJevEvaluatorOptions<
  RecordType extends Record<string, unknown>,
> = {
  name: string;
  choices: ClassificationChoicesMap;
  buildRequest: (record: RecordType) => JevQuestionRequest;
  model?: Experimental_EvaluationModel;
  modelId?: string;
  tracer?: Tracer;
};

function getConfidence(providerMetadata: unknown): number | undefined {
  if (
    typeof providerMetadata !== "object" ||
    providerMetadata === null ||
    !("typesafe" in providerMetadata)
  ) {
    return undefined;
  }
  const typesafe = providerMetadata.typesafe;
  if (
    typeof typesafe !== "object" ||
    typesafe === null ||
    !("confidence" in typesafe)
  ) {
    return undefined;
  }
  const confidence = typesafe.confidence;
  if (
    typeof confidence !== "object" ||
    confidence === null ||
    !("label" in confidence)
  ) {
    return undefined;
  }
  return typeof confidence.label === "number" ? confidence.label : undefined;
}

function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function setUsageAttributes({
  span,
  inputTokens,
  outputTokens,
  totalTokens,
}: {
  span: Span;
  inputTokens: number | undefined;
  outputTokens: number | undefined;
  totalTokens: number | undefined;
}): void {
  if (inputTokens !== undefined) {
    span.setAttribute("llm.token_count.prompt", inputTokens);
  }
  if (outputTokens !== undefined) {
    span.setAttribute("llm.token_count.completion", outputTokens);
  }
  if (totalTokens !== undefined) {
    span.setAttribute("llm.token_count.total", totalTokens);
  }
}

export function createJevEvaluator<
  RecordType extends Record<string, unknown> = Record<string, unknown>,
>({
  name,
  choices,
  buildRequest,
  modelId = DEFAULT_MODEL_ID,
  model = typeSafeAi.evaluationModel(modelId),
  tracer,
}: CreateJevEvaluatorOptions<RecordType>): JevEvaluator<RecordType> {
  const runJev = async (record: RecordType): Promise<JevEvaluationResult> => {
    const request = buildRequest(record);
    const questions = {
      label: {
        type: "choice" as const,
        instructions: request.instructions,
        criteria: request.criteria,
      },
    };

    const invocationTracer = tracer ?? trace.getTracer("judge-sweep");
    return invocationTracer.startActiveSpan(`${name}.jev`, async (span) => {
      span.setAttributes({
        "openinference.span.kind": "LLM",
        "llm.request.model_name": modelId,
        "llm.provider": "typesafe",
        "input.value": JSON.stringify({ state: request.state, questions }),
        "input.mime_type": JSON_MIME_TYPE,
      });

      try {
        const response = await experimental_evaluate({
          model,
          state: request.state,
          questions,
        });
        const answer = response.answers.label;
        if (answer.type !== "choice") {
          throw new Error(`Jev returned ${answer.type} for a choice question`);
        }
        if (!Object.hasOwn(choices, answer.choice)) {
          throw new Error(
            `Jev returned invalid label ${JSON.stringify(answer.choice)} for ${name}`
          );
        }

        const resolvedModel = response.response.modelId;
        const inputTokens = response.usage.inputTokens;
        const outputTokens = response.usage.outputTokens;
        const totalTokens = response.usage.totalTokens;
        const result: JevEvaluationResult = {
          label: answer.choice,
          score: choices[answer.choice] as number,
          metadata: {
            model: modelId,
            resolvedModel,
            probabilities: answer.probabilities ?? {},
            confidence: getConfidence(response.providerMetadata),
            usage: { inputTokens, outputTokens, totalTokens },
          },
        };

        span.setAttribute("llm.model_name", resolvedModel);
        setUsageAttributes({ span, inputTokens, outputTokens, totalTokens });
        span.setAttributes({
          "output.value": JSON.stringify({
            answers: response.answers,
            usage: response.usage,
            modelId: resolvedModel,
          }),
          "output.mime_type": JSON_MIME_TYPE,
        });
        span.setStatus({ code: SpanStatusCode.OK });
        return result;
      } catch (error) {
        span.recordException(getErrorMessage(error));
        span.setStatus({
          code: SpanStatusCode.ERROR,
          message: getErrorMessage(error),
        });
        throw error;
      } finally {
        span.end();
      }
    });
  };

  return {
    name,
    kind: "LLM",
    async evaluate(record) {
      let detailedResult: JevEvaluationResult | undefined;
      const phoenixEvaluator = createEvaluator<RecordType>(
        async () => {
          detailedResult = await runJev(record);
          return detailedResult;
        },
        { name, kind: "LLM" }
      );
      await phoenixEvaluator.evaluate(record);
      if (detailedResult === undefined) {
        throw new Error(`${name} completed without an evaluation result`);
      }
      return detailedResult;
    },
  };
}
