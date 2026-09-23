import { defineConfig } from "vitest/config";

import { getMaxConcurrentCallsPerSuite } from "./src/judges.js";

export default defineConfig({
  test: {
    include: ["src/**/*.eval.?(c|m)[jt]s"],
    reporters: ["default", "@arizeai/phoenix-client/vitest/reporter"],
    setupFiles: ["dotenv/config"],
    testTimeout: 120_000,
    fileParallelism: true,
    sequence: { concurrent: true },
    maxConcurrency: getMaxConcurrentCallsPerSuite(),
  },
});
