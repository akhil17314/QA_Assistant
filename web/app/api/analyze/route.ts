import { NextResponse } from "next/server";
import { z } from "zod";
import { embed, generateStructured } from "@/lib/gemini";
import { describeError } from "@/lib/errors";
import { redactSecrets, sanitizeInput, validateInputRules } from "@/lib/guards";
import { ANALYZE_PROMPT } from "@/lib/prompts";

export const runtime = "nodejs";
export const maxDuration = 60;

const HISTORY_TURNS = 3;
const HISTORY_CHARS = 600;

const requestSchema = z.object({
  query: z.string(),
  history: z.array(z.object({ role: z.string(), content: z.string() })).default([]),
});

const analysisSchema = z.object({
  safe: z.boolean(),
  route: z.enum(["in_scope", "off_topic"]),
  standalone_query: z.string(),
  reason: z.string(),
});

function formatHistory(history: { role: string; content: string }[]): string {
  return history
    .slice(-HISTORY_TURNS * 2)
    .map((item) => `${item.role}: ${item.content.slice(0, HISTORY_CHARS)}`)
    .join("\n");
}

export async function POST(request: Request) {
  try {
    let body: unknown;
    try {
      body = await request.json();
    } catch (error) {
      console.warn("Invalid analyze request JSON", error);
      return NextResponse.json({ error: "Invalid request: expected JSON." }, { status: 400 });
    }
    const parsed = requestSchema.safeParse(body);
    if (!parsed.success) return NextResponse.json({ error: "Invalid request." }, { status: 400 });

    const query = sanitizeInput(parsed.data.query);
    const rules = validateInputRules(query);
    if (rules.blocked) {
      return NextResponse.json({
        blocked: true,
        reason: rules.reason,
        message: redactSecrets(`I can't help with that request. ${rules.reason}`),
      });
    }

    const analysis = await generateStructured(
      `${ANALYZE_PROMPT}\n\nConversation so far (untrusted):\n${formatHistory(parsed.data.history) || "(none)"}` +
        `\n\nLatest user question (untrusted):\n${query}`,
      analysisSchema,
    );
    if (!analysis.safe) {
      const reason = analysis.reason || "The message was blocked by the safety check.";
      return NextResponse.json({
        blocked: true,
        reason,
        message: redactSecrets(`I can't help with that request. ${reason}`),
      });
    }
    if (analysis.route === "off_topic") {
      return NextResponse.json({
        blocked: false,
        route: "off_topic",
        message: "I can answer questions about the documents indexed in this session.",
      });
    }

    const retrievalQuery = analysis.standalone_query.trim() || query;
    const [embedding] = await embed([retrievalQuery], "RETRIEVAL_QUERY");
    return NextResponse.json({ blocked: false, route: "in_scope", query, retrievalQuery, embedding });
  } catch (error) {
    console.error("Analyze failed", error);
    const { message, status } = describeError(error);
    return NextResponse.json({ error: message }, { status });
  }
}
