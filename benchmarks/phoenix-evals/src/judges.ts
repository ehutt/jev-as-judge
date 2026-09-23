import { anthropic } from "@ai-sdk/anthropic";
import { google } from "@ai-sdk/google";
import { openai } from "@ai-sdk/openai";
import type { LanguageModel } from "ai";

export type ConcreteLanguageModel = Exclude<LanguageModel, string>;

export const PRICING_DATE = "2026-09-21";
export const DEFAULT_MAX_CONCURRENT_CALLS_PER_SUITE = 5;

export const JUDGES = {
  "jev-slot-in": {
    provider: "typesafe",
    maxConcurrentCallsPerSuite: 1,
    modelId: "jev-latest",
    variant: "slot-in",
    pricing: { input: 0.042, output: 0 },
    pricingSource: "https://docs.typesafe.ai/models",
    apiKeyEnvironmentVariable: "TYPESAFE_AI_API_KEY",
  },
  "jev-native": {
    provider: "typesafe",
    maxConcurrentCallsPerSuite: 1,
    modelId: "jev-latest",
    variant: "native",
    pricing: { input: 0.042, output: 0 },
    pricingSource: "https://docs.typesafe.ai/models",
    apiKeyEnvironmentVariable: "TYPESAFE_AI_API_KEY",
  },
  "openai-cheap": {
    provider: "openai",
    maxConcurrentCallsPerSuite: 10,
    modelId: "gpt-5-nano",
    variant: "llm",
    pricing: { input: 0.05, output: 0.4 },
    pricingSource: "https://developers.openai.com/api/docs/pricing",
    apiKeyEnvironmentVariable: "OPENAI_API_KEY",
  },
  "openai-frontier": {
    provider: "openai",
    maxConcurrentCallsPerSuite: 10,
    modelId: "gpt-5.6-sol",
    variant: "llm",
    pricing: { input: 4, output: 20 },
    pricingSource: "https://developers.openai.com/api/docs/pricing",
    apiKeyEnvironmentVariable: "OPENAI_API_KEY",
  },
  "anthropic-cheap": {
    provider: "anthropic",
    maxConcurrentCallsPerSuite: 10,
    modelId: "claude-haiku-4-5",
    variant: "llm",
    pricing: { input: 1, output: 5 },
    pricingSource: "https://docs.anthropic.com/en/docs/about-claude/pricing",
    apiKeyEnvironmentVariable: "ANTHROPIC_API_KEY",
  },
  "anthropic-frontier": {
    provider: "anthropic",
    maxConcurrentCallsPerSuite: 10,
    modelId: "claude-opus-5",
    variant: "llm",
    pricing: { input: 5, output: 25 },
    pricingSource: "https://docs.anthropic.com/en/docs/about-claude/pricing",
    apiKeyEnvironmentVariable: "ANTHROPIC_API_KEY",
  },
  gemini: {
    provider: "google",
    maxConcurrentCallsPerSuite: 10,
    modelId: "gemini-3.8-flash",
    variant: "llm",
    pricing: { input: 0.75, output: 3.75 },
    pricingSource: "https://ai.google.dev/gemini-api/docs/pricing",
    apiKeyEnvironmentVariable: "GOOGLE_GENERATIVE_AI_API_KEY",
  },
} as const;

export type JudgeId = keyof typeof JUDGES;
export type Judge = (typeof JUDGES)[JudgeId];

export const JUDGE_IDS = Object.keys(JUDGES) as JudgeId[];

export function isJudgeId(value: string): value is JudgeId {
  return Object.hasOwn(JUDGES, value);
}

export function getJudge(id: string | undefined = process.env.JUDGE): Judge {
  if (id === undefined) {
    throw new Error(
      `JUDGE is required. Choose one of: ${JUDGE_IDS.join(", ")}`
    );
  }
  if (!isJudgeId(id)) {
    throw new Error(
      `Unknown judge ${JSON.stringify(id)}. Choose one of: ${JUDGE_IDS.join(", ")}`
    );
  }
  return JUDGES[id];
}

export function getJudgeId(
  value: string | undefined = process.env.JUDGE
): JudgeId {
  getJudge(value);
  return value as JudgeId;
}

export function getMaxConcurrentCallsPerSuite(): number {
  const override = process.env.SWEEP_CONCURRENCY;
  if (override !== undefined) {
    const parsed = Number.parseInt(override, 10);
    if (!Number.isInteger(parsed) || parsed < 1) {
      throw new Error(
        `SWEEP_CONCURRENCY must be a positive integer, received ${JSON.stringify(override)}`
      );
    }
    return parsed;
  }
  const judgeId = process.env.JUDGE;
  return judgeId !== undefined && isJudgeId(judgeId)
    ? JUDGES[judgeId].maxConcurrentCallsPerSuite
    : DEFAULT_MAX_CONCURRENT_CALLS_PER_SUITE;
}

export function createLanguageModel(judgeId: JudgeId): ConcreteLanguageModel {
  const judge = JUDGES[judgeId];
  switch (judge.provider) {
    case "openai":
      return openai(judge.modelId);
    case "anthropic":
      return anthropic(judge.modelId);
    case "google":
      return google(judge.modelId);
    case "typesafe":
      throw new Error(
        `${judgeId} is an evaluation model, not a language model`
      );
  }
}
