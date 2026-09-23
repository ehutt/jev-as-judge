import { trace } from "@opentelemetry/api";
import {
  BasicTracerProvider,
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { Experimental_EvaluationMockModelV4 } from "ai/test";
import { describe, expect, it } from "vitest";

import { createJevEvaluator } from "./createJevEvaluator.js";

function createRecorder() {
  const exporter = new InMemorySpanExporter();
  const provider = new BasicTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(exporter)],
  });
  return { exporter, tracer: provider.getTracer("test") };
}

describe("createJevEvaluator", () => {
  it("maps a choice response and records the pricing attributes", async () => {
    const { exporter, tracer } = createRecorder();
    const model = new Experimental_EvaluationMockModelV4({
      modelId: "jev-latest",
      doEvaluate: async () => ({
        answers: {
          label: {
            type: "choice",
            choice: "correct",
            probabilities: { correct: 0.8, incorrect: 0.2 },
          },
        },
        usage: { inputTokens: 120, outputTokens: 0 },
        warnings: [],
        providerMetadata: {
          typesafe: { confidence: { label: 0.72 } },
        },
        response: { modelId: "jev-1.13.0" },
      }),
    });
    const evaluator = createJevEvaluator({
      name: "correctness",
      choices: { correct: 1, incorrect: 0 },
      model,
      tracer,
      buildRequest: (record: { input: string; output: string }) => ({
        state: record,
        instructions: "Classify correctness.",
        criteria: { correct: "Correct", incorrect: "Incorrect" },
      }),
    });

    const result = await evaluator.evaluate({
      input: "What is 2 + 2?",
      output: "4",
    });

    expect(result).toEqual({
      label: "correct",
      score: 1,
      metadata: {
        model: "jev-latest",
        resolvedModel: "jev-1.13.0",
        probabilities: { correct: 0.8, incorrect: 0.2 },
        confidence: 0.72,
        usage: { inputTokens: 120, outputTokens: 0, totalTokens: 120 },
      },
    });
    expect(result).not.toHaveProperty("explanation");

    const [span] = exporter.getFinishedSpans();
    expect(span?.attributes).toMatchObject({
      "openinference.span.kind": "LLM",
      "llm.request.model_name": "jev-latest",
      "llm.model_name": "jev-1.13.0",
      "llm.provider": "typesafe",
      "llm.token_count.prompt": 120,
      "llm.token_count.completion": 0,
      "llm.token_count.total": 120,
      "input.mime_type": "application/json",
      "output.mime_type": "application/json",
    });
  });

  it("rejects labels outside the evaluator choices", async () => {
    const { tracer } = createRecorder();
    const model = new Experimental_EvaluationMockModelV4({
      doEvaluate: async () => ({
        answers: {
          label: {
            type: "choice",
            choice: "maybe",
            probabilities: { maybe: 1 },
          },
        },
        usage: { inputTokens: 10, outputTokens: 0 },
        warnings: [],
      }),
    });
    const evaluator = createJevEvaluator({
      name: "correctness",
      choices: { correct: 1, incorrect: 0 },
      model,
      tracer,
      buildRequest: () => ({
        state: "state",
        instructions: "Classify correctness.",
        criteria: { correct: null, incorrect: null },
      }),
    });

    await expect(evaluator.evaluate({})).rejects.toThrow(
      'Question "label" selected an unknown option.'
    );
  });

  it("resolves the default tracer after the evaluator is created", async () => {
    const model = new Experimental_EvaluationMockModelV4({
      doEvaluate: async () => ({
        answers: {
          label: {
            type: "choice",
            choice: "correct",
            probabilities: { correct: 1, incorrect: 0 },
          },
        },
        usage: { inputTokens: 10, outputTokens: 2 },
        warnings: [],
      }),
    });
    const evaluator = createJevEvaluator({
      name: "correctness",
      choices: { correct: 1, incorrect: 0 },
      model,
      buildRequest: () => ({
        state: "state",
        instructions: "Classify correctness.",
        criteria: { correct: null, incorrect: null },
      }),
    });
    const exporter = new InMemorySpanExporter();
    const provider = new BasicTracerProvider({
      spanProcessors: [new SimpleSpanProcessor(exporter)],
    });

    trace.setGlobalTracerProvider(provider);
    try {
      await evaluator.evaluate({});
      expect(
        exporter
          .getFinishedSpans()
          .some((span) => span.name === "correctness.jev")
      ).toBe(true);
    } finally {
      trace.disable();
      await provider.shutdown();
    }
  });
});
