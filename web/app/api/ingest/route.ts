import { NextResponse } from "next/server";
import { extractText, getDocumentProxy } from "unpdf";
import { chunkText } from "@/lib/chunk";
import { embed } from "@/lib/gemini";
import { describeError } from "@/lib/errors";

export const runtime = "nodejs";
export const maxDuration = 60;

const MAX_PDF_PAGES = 300;
// Vercel rejects request bodies over 4.5 MB.
const MAX_UPLOAD_BYTES = 4 * 1024 * 1024;

export async function POST(request: Request) {
  try {
    const form = await request.formData();
    const file = form.get("file");
    if (!(file instanceof File)) return NextResponse.json({ error: "No file uploaded." }, { status: 400 });
    if (file.size > MAX_UPLOAD_BYTES) {
      return NextResponse.json({ error: "File exceeds the 4 MB upload limit." }, { status: 413 });
    }

    const source = file.name.split(/[\\/]/).pop() || "document.pdf";
    let pages: string[];
    try {
      const pdf = await getDocumentProxy(new Uint8Array(await file.arrayBuffer()));
      pages = (await extractText(pdf, { mergePages: false })).text;
    } catch (error) {
      console.warn("PDF parsing failed", error);
      return NextResponse.json({ error: "The file could not be read as a valid PDF." }, { status: 400 });
    }

    const nonEmpty = pages.filter((text) => text.trim());
    if (!nonEmpty.length) {
      return NextResponse.json(
        { error: "No extractable text found (scanned or image-only PDFs are unsupported)." },
        { status: 422 },
      );
    }
    if (nonEmpty.length > MAX_PDF_PAGES) {
      return NextResponse.json(
        { error: `PDF has ${nonEmpty.length} text pages; the limit is ${MAX_PDF_PAGES}.` },
        { status: 422 },
      );
    }

    const chunks = pages.flatMap((text, index) =>
      chunkText(text.trim()).map((chunk) => ({ text: chunk, source, page: index + 1 })),
    );
    const vectors = await embed(
      chunks.map((chunk) => chunk.text),
      "RETRIEVAL_DOCUMENT",
    );
    return NextResponse.json({
      source,
      chunks: chunks.map((chunk, index) => ({ ...chunk, embedding: vectors[index] })),
    });
  } catch (error) {
    console.error("Ingest failed", error);
    const { message, status } = describeError(error);
    return NextResponse.json({ error: message }, { status });
  }
}
