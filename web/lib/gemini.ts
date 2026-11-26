import { GoogleGenAI } from "@google/genai";
import { z } from "zod";
import { ConfigurationError } from "./errors";

export const LLM_MODEL = process.env.LLM_MODEL || "gemini-2.5-flash";
export const JUDGE_MODEL = process.env.JUDGE_MODEL || LLM_MODEL;
export const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL || "gemini-embedding-001";
export const EMBEDDING_DIMENSIONS = 768;

let client: GoogleGenAI | null = null;

function getClient(): GoogleGenAI {
  const apiKey = process.env.GOOGLE_API_KEY;
  if (!apiKey) throw new ConfigurationError("GOOGLE_API_KEY is not configured on the server.");
  client ??= new GoogleGenAI({ apiKey });
  return client;
}

export async function generateStructured<T extends z.ZodType>(
  prompt: string,
  schema: T,
  model: string = LLM_MODEL,
): Promise<z.infer<T>> {
  const response = await getClient().models.generateContent({
    model,
    contents: prompt,
    config: {
      temperature: 0,
      responseMimeType: "application/json",
      responseJsonSchema: z.toJSONSchema(schema),
    },
  });
  return schema.parse(JSON.parse(response.text ?? ""));
}

function normalize(vector: number[]): number[] {
  const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0)) || 1;
  return vector.map((value) => Math.round((value / norm) * 1e4) / 1e4);
}

export async function embed(texts: string[], task: "RETRIEVAL_DOCUMENT" | "RETRIEVAL_QUERY") {
  const vectors: number[][] = [];
  for (let start = 0; start < texts.length; start += 100) {
    const response = await getClient().models.embedContent({
      model: EMBEDDING_MODEL,
      contents: texts.slice(start, start + 100),
      config: { taskType: task, outputDimensionality: EMBEDDING_DIMENSIONS },
    });
    for (const item of response.embeddings ?? []) vectors.push(normalize(item.values ?? []));
  }
  if (vectors.length !== texts.length) throw new Error("Embedding service returned an unexpected result.");
  return vectors;
}
