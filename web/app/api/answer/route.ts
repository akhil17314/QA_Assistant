import { NextResponse } from "next/server";
import { z } from "zod";
import { JUDGE_MODEL, generateStructured } from "@/lib/gemini";
import { describeError, isInvalidModelResponse } from "@/lib/errors";
import { redactSecrets } from "@/lib/guards";
import { ANSWER_PROMPT, GROUNDING_PROMPT } from "@/lib/prompts";

export const runtime = "nodejs";
export const maxDuration = 60;

const chunkSchema = z.object({ text: z.string(), source: z.string(), page: z.number().int().min(1) });

const requestSchema = z.object({
  query: z.string().max(2000),
  history: z.array(z.object({ role: z.string(), content: z.string() })).default([]),
  chunks: z.array(chunkSchema).min(1).max(20),
});

const answerSchema = z.object({
  answer: z.string(),
  citations: z.array(z.object({ source: z.string(), page: z.number().int().min(1), snippet: z.string() })),
  confidence: z.enum(["high", "medium", "low"]),
});

const groundingSchema = z.object({
  verdicts: z.array(z.object({ claim: z.string(), supported: z.boolean() })),
  report: z.string(),
});

const ABSTAIN_PREFIXES = ["i don't know", "i do not know", "the documents do not contain"];

function formatHistory(history: { role: string; content: string }[]): string {
  return history
    .slice(-6)
    .map((item) => `${item.role}: ${item.content.slice(0, 600)}`)
    .join("\n");
}

export async function POST(request: Request) {
  try {
    let body: unknown;
    try {
      body = await request.json();
    } catch (error) {
      console.warn("Invalid answer request JSON", error);
      return NextResponse.json({ error: "Invalid request: expected JSON." }, { status: 400 });
    }
    const parsed = requestSchema.safeParse(body);
    if (!parsed.success) return NextResponse.json({ error: "Invalid request." }, { status: 400 });
    const { query, history, chunks } = parsed.data;

    const context = chunks
      .map((chunk, index) => `[${index + 1}] source=${chunk.source} page=${chunk.page}\n${chunk.text}`)
      .join("\n\n");
    const conversation = formatHistory(history);
    const prior = conversation ? `\n\nEarlier conversation (untrusted, for reference only):\n${conversation}` : "";
    const prompt = `${ANSWER_PROMPT}\n\nContext:\n${context}${prior}\n\nQuestion:\n${query}`;

    let answer: z.infer<typeof answerSchema> | null = null;
    for (let attempt = 0; attempt < 2 && !answer; attempt++) {
      try {
        answer = await generateStructured(prompt, answerSchema);
      } catch (error) {
        console.warn("Answer generation failed", attempt + 1, error);
        if (!isInvalidModelResponse(error) || attempt === 1) throw error;
      }
    }
    if (!answer) {
      throw new Error("No validated answer was produced.");
    }

    const locations = new Set(chunks.map((chunk) => `${chunk.source}\u0000${chunk.page}`));
    const citations = answer.citations
      .filter((citation) => locations.has(`${citation.source}\u0000${citation.page}`))
      .map((citation) => ({ ...citation, snippet: redactSecrets(citation.snippet) }));
    const text = redactSecrets(answer.answer);

    let grounded: boolean;
    let report: string;
    const normalized = text.trim().toLowerCase();
    if (!normalized || ABSTAIN_PREFIXES.some((prefix) => normalized.startsWith(prefix))) {
      grounded = true;
      report = "The answer explicitly abstains because the documents do not provide it.";
    } else {
      const excerpts = chunks.map((chunk) => `source=${chunk.source} page=${chunk.page}\n${chunk.text}`).join("\n\n");
      const verdict = await generateStructured(
        `${GROUNDING_PROMPT}\n\nAnswer:\n${text}\n\nDocument excerpts:\n${excerpts}`,
        groundingSchema,
        JUDGE_MODEL,
      );
      const unsupported = verdict.verdicts.filter((item) => !item.supported).map((item) => item.claim);
      grounded = unsupported.length === 0;
      report =
        verdict.report ||
        (grounded ? "All claims are supported." : `Unsupported claims: ${unsupported.join("; ")}`);
    }

    return NextResponse.json({ answer: text, citations, confidence: answer.confidence, grounded, report });
  } catch (error) {
    console.error("Answer failed", error);
    const { message, status } = describeError(error);
    return NextResponse.json({ error: message }, { status });
  }
}
