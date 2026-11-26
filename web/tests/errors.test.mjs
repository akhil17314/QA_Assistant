import assert from "node:assert/strict";
import test from "node:test";
import { ApiError } from "@google/genai";
import { z } from "zod";
import { ConfigurationError, describeError, isInvalidModelResponse } from "../lib/errors.ts";

test("reports a Gemini overload without exposing the raw response", () => {
  const error = new ApiError({
    status: 503,
    message: JSON.stringify({ error: { code: 503, message: "private diagnostic", status: "UNAVAILABLE" } }),
  });
  const result = describeError(error);
  assert.equal(result.status, 503);
  assert.match(result.message, /high demand.*503.*try again/i);
  assert.doesNotMatch(result.message, /private diagnostic/);
  assert.equal(isInvalidModelResponse(error), false);
});

test("reports model and API key errors with actionable guidance", () => {
  assert.match(describeError(new ApiError({ status: 404, message: "not found" })).message, /model.*404/i);
  assert.match(describeError(new ApiError({ status: 400, message: "API_KEY_INVALID" })).message, /API key/i);
  assert.match(describeError(new ConfigurationError("GOOGLE_API_KEY is not configured on the server.")).message, /GOOGLE_API_KEY/);
});

test("retries malformed model responses, not provider failures", () => {
  const invalid = z.object({ answer: z.string() }).safeParse({});
  assert.equal(invalid.success, false);
  if (!invalid.success) {
    assert.equal(isInvalidModelResponse(invalid.error), true);
    assert.equal(describeError(invalid.error).status, 502);
  }
  assert.equal(isInvalidModelResponse(new SyntaxError("bad JSON")), true);
  assert.equal(isInvalidModelResponse(new ApiError({ status: 429, message: "quota" })), false);
});

test("does not disclose unexpected server errors", () => {
  const result = describeError(new Error("secret internal details"));
  assert.equal(result.status, 500);
  assert.doesNotMatch(result.message, /secret internal details/);
});
