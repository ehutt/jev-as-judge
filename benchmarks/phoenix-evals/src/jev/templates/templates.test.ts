import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import {
  EVALUATOR_NAMES,
  SOURCE_FILE_BY_EVALUATOR,
  SOURCE_SHA256_BY_EVALUATOR,
  TEMPLATE_SOURCE_SHA,
} from "./index.js";

const localDirectory = fileURLToPath(new URL("./", import.meta.url));

describe("copied evaluator templates", () => {
  it("records the source commit", () => {
    expect(TEMPLATE_SOURCE_SHA).toBe(
      "6f03f903b8d3eddffe11e6b695c1c24e244f946a"
    );
  });

  for (const evaluatorName of EVALUATOR_NAMES) {
    it(`${evaluatorName} matches its recorded source digest`, () => {
      const fileName = SOURCE_FILE_BY_EVALUATOR[evaluatorName];
      const copy = readFileSync(join(localDirectory, fileName), "utf8");
      const digest = createHash("sha256").update(copy).digest("hex");
      expect(digest).toBe(SOURCE_SHA256_BY_EVALUATOR[evaluatorName]);
    });
  }
});
