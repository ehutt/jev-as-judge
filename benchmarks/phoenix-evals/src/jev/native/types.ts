import type { JSONValue, ModelMessage } from "ai";

import type { ClassificationEvaluatorConfig } from "../templates/types.js";

export type JevQuestionRequest = {
  state: string | Readonly<Record<string, JSONValue>> | readonly JSONValue[];
  instructions:
    | string
    | Readonly<Record<string, JSONValue>>
    | readonly JSONValue[];
  criteria: Readonly<
    Record<
      string,
      string | Readonly<Record<string, JSONValue>> | readonly JSONValue[] | null
    >
  >;
};

export type NativeQuestionSpec<
  RecordType extends Record<string, unknown> = Record<string, unknown>,
> = {
  config: ClassificationEvaluatorConfig;
  buildState: (
    record: RecordType
  ) => Readonly<Record<string, JSONValue>> | readonly JSONValue[];
  criteria: JevQuestionRequest["criteria"];
};

/** The prompt as plain text: message contents joined, no chat-message wrapper. */
export function flattenTemplate(template: string | ModelMessage[]): string {
  if (typeof template === "string") {
    return template;
  }
  return template
    .map((message) =>
      typeof message.content === "string"
        ? message.content
        : JSON.stringify(message.content)
    )
    .join("\n");
}

function getRubricInstructions(config: ClassificationEvaluatorConfig): string {
  return flattenTemplate(config.template)
    .replace(/\n?<data>[\s\S]*?<\/data>\n?/u, "\n")
    .trim();
}

export function buildNativeRequest<RecordType extends Record<string, unknown>>({
  spec,
  record,
}: {
  spec: NativeQuestionSpec<RecordType>;
  record: RecordType;
}): JevQuestionRequest {
  return {
    state: spec.buildState(record),
    instructions: getRubricInstructions(spec.config),
    criteria: spec.criteria,
  };
}
