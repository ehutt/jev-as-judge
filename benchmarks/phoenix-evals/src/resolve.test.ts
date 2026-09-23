import { describe, expect, it } from "vitest";

import { getSlotInRequest } from "./resolve.js";

describe("getSlotInRequest", () => {
  it("renders the built-in prompt as plain text state", () => {
    const request = getSlotInRequest({
      evaluatorName: "hallucination",
      record: { input: "User: What is 2 + 2?", output: "4" },
    });

    expect(typeof request.state).toBe("string");
    const state = request.state as string;
    expect(state).toContain("<input>\nUser: What is 2 + 2?\n</input>");
    expect(state).toContain("<output>\n4\n</output>");
    expect(state).not.toContain("{{");
    expect(state).not.toContain('"role"');
    expect(request.criteria).toEqual({ hallucinated: null, grounded: null });
  });
});
