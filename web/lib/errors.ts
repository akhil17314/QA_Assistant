import { ApiError } from "@google/genai";
import { ZodError } from "zod";

export class ConfigurationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ConfigurationError";
  }
}

export function isInvalidModelResponse(error: unknown): boolean {
  return error instanceof ZodError || error instanceof SyntaxError;
}

export function describeError(error: unknown): { message: string; status: number } {
  if (error instanceof ConfigurationError) {
    return { message: error.message, status: 500 };
  }
  if (error instanceof ApiError) {
    const status = error.status;
    const message = error.message.toLowerCase();
    if (["api_key_invalid", "api key not valid", "invalid api key", "api key expired"].some((marker) => message.includes(marker))) {
      return { message: "Gemini rejected the API key. Check GOOGLE_API_KEY on the server.", status: 502 };
    }
    switch (status) {
      case 400:
        return { message: "Gemini rejected the request (400). Check the model configuration and request.", status: 502 };
      case 401:
        return { message: "Gemini authentication failed (401). Check GOOGLE_API_KEY on the server.", status: 502 };
      case 403:
        return { message: "Gemini denied access (403). Check the API key, project permissions, and model access.", status: 502 };
      case 404:
        return { message: "Gemini model not found or unavailable (404). Check LLM_MODEL, JUDGE_MODEL, or EMBEDDING_MODEL on the server.", status: 502 };
      case 429:
        return { message: "Gemini rate limit or quota exceeded (429). Please try again later or check your quota.", status: 429 };
      case 500:
        return { message: "Gemini encountered an internal error (500). Please try again later.", status: 502 };
      case 503:
        return { message: "Gemini is currently unavailable or experiencing high demand (503). Please try again later.", status: 503 };
      case 504:
        return { message: "Gemini timed out (504). Please try again later.", status: 504 };
      default:
        return {
          message: `Gemini request failed${status ? ` (${status})` : ""}. Check the server logs for details.`,
          status: 502,
        };
    }
  }
  if (isInvalidModelResponse(error)) {
    return { message: "Gemini returned an invalid response. Please try again.", status: 502 };
  }
  return { message: "An unexpected server error occurred. Check the server logs for details.", status: 500 };
}
