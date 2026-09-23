import { config as loadEnvironment } from "dotenv";

import { JUDGES, type Judge } from "./src/judges.js";

loadEnvironment({ quiet: true });

type TokenPrice = {
  tokenType: string;
  kind: "PROMPT" | "COMPLETION";
  costPerMillionTokens: number;
};

type GenerativeModel = {
  id: string;
  name: string;
  provider: string | null;
  namePattern: string;
  kind: "BUILT_IN" | "CUSTOM";
  tokenPrices: TokenPrice[];
};

const MODELS_QUERY = `
  query JudgeSweepModels {
    generativeModels(first: 1000) {
      edges {
        node {
          id
          name
          provider
          namePattern
          kind
          tokenPrices {
            tokenType
            kind
            costPerMillionTokens
          }
        }
      }
    }
  }
`;

const CREATE_MODEL_MUTATION = `
  mutation JudgeSweepCreateModel($input: CreateModelMutationInput!) {
    createModel(input: $input) {
      model { id name provider namePattern }
    }
  }
`;

function getBaseUrl(): string {
  const configured =
    process.env.PHOENIX_ENDPOINT ??
    process.env.PHOENIX_COLLECTOR_ENDPOINT ??
    "http://localhost:6006";
  return configured
    .replace(/\/v1\/(?:traces|logs|metrics)\/?$/u, "")
    .replace(/\/$/u, "");
}

function getHeaders(): Record<string, string> {
  const configuredHeaders = process.env.PHOENIX_CLIENT_HEADERS;
  const headers: Record<string, string> = configuredHeaders
    ? (JSON.parse(configuredHeaders) as Record<string, string>)
    : {};
  if (process.env.PHOENIX_API_KEY) {
    headers.Authorization = `Bearer ${process.env.PHOENIX_API_KEY}`;
  }
  return { ...headers, "content-type": "application/json" };
}

async function graphql<Data>({
  query,
  variables,
}: {
  query: string;
  variables?: Record<string, unknown>;
}): Promise<Data> {
  const response = await fetch(`${getBaseUrl()}/graphql`, {
    method: "POST",
    headers: getHeaders(),
    body: JSON.stringify({ query, variables }),
  });
  if (!response.ok) {
    throw new Error(
      `Phoenix GraphQL returned HTTP ${response.status}: ${await response.text()}`
    );
  }
  const body = (await response.json()) as {
    data?: Data;
    errors?: Array<{ message: string }>;
  };
  if (body.errors?.length) {
    throw new Error(body.errors.map((error) => error.message).join("\n"));
  }
  if (body.data === undefined) {
    throw new Error("Phoenix GraphQL response did not include data");
  }
  return body.data;
}

function doesModelMatch({
  model,
  judge,
}: {
  model: GenerativeModel;
  judge: Judge;
}): boolean {
  const providerMatches =
    model.provider === null ||
    model.provider === "" ||
    model.provider === judge.provider;
  if (!providerMatches) {
    return false;
  }
  try {
    return new RegExp(model.namePattern).test(judge.modelId);
  } catch {
    return false;
  }
}

function escapeRegularExpression(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/gu, "\\$&");
}

function getNamePattern(judge: Judge): string {
  return judge.provider === "typesafe"
    ? "^jev(?:-.+)?$"
    : `^${escapeRegularExpression(judge.modelId)}(?:-.+)?$`;
}

async function listModels(): Promise<GenerativeModel[]> {
  const data = await graphql<{
    generativeModels: { edges: Array<{ node: GenerativeModel }> };
  }>({ query: MODELS_QUERY });
  return data.generativeModels.edges.map((edge) => edge.node);
}

async function createPricingModel(judge: Judge): Promise<void> {
  const normalizedModelName = judge.modelId.replace(/[^a-zA-Z0-9_-]/gu, "-");
  await graphql({
    query: CREATE_MODEL_MUTATION,
    variables: {
      input: {
        name: `judge-sweep-${judge.provider}-${normalizedModelName}`,
        provider: judge.provider,
        namePattern: getNamePattern(judge),
        costs: [
          {
            tokenType: "input",
            kind: "PROMPT",
            costPerMillionTokens: judge.pricing.input,
          },
          {
            tokenType: "output",
            kind: "COMPLETION",
            costPerMillionTokens: judge.pricing.output,
          },
        ],
      },
    },
  });
  process.stdout.write(`registered ${judge.provider}/${judge.modelId}\n`);
}

async function main(): Promise<void> {
  const models = await listModels();
  const uniqueJudges = Array.from(
    new Map(
      Object.values(JUDGES).map((judge) => [
        `${judge.provider}/${judge.modelId}`,
        judge,
      ])
    ).values()
  );

  for (const judge of uniqueJudges) {
    const matchingModel = models.find((model) =>
      doesModelMatch({ model, judge })
    );
    if (matchingModel) {
      process.stdout.write(
        `verified ${judge.provider}/${judge.modelId} via ${matchingModel.name} (${matchingModel.kind})\n`
      );
      continue;
    }
    await createPricingModel(judge);
  }
}

await main();
