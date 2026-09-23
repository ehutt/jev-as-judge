import type { ModelMessage } from "ai";

export type ClassificationEvaluatorConfig = {
  name: string;
  description: string;
  optimizationDirection: "MINIMIZE" | "MAXIMIZE" | "NEUTRAL";
  template: string | ModelMessage[];
  choices: Record<string, number>;
};
