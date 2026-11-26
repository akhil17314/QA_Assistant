export const CHUNK_SIZE = 500;
export const CHUNK_OVERLAP = 50;

const SEPARATORS = ["\n\n", "\n", " ", ""];

function split(text: string, size: number, separators: string[]): string[] {
  if (text.length <= size) return [text];
  const [separator, ...rest] = separators;
  const parts = separator === "" ? Array.from(text) : text.split(separator);
  const pieces: string[] = [];
  for (const part of parts) {
    if (part.length > size && rest.length) pieces.push(...split(part, size, rest));
    else pieces.push(part);
  }
  return pieces;
}

/** Recursive character splitter: merge small pieces up to `size`, carrying `overlap` characters forward. */
export function chunkText(text: string, size = CHUNK_SIZE, overlap = CHUNK_OVERLAP): string[] {
  const pieces = split(text, size, SEPARATORS);
  const chunks: string[] = [];
  let current: string[] = [];
  let length = 0;
  const join = (items: string[]) => items.join(" ").trim();

  for (const piece of pieces) {
    if (current.length && length + piece.length + 1 > size) {
      chunks.push(join(current));
      while (current.length && (length > overlap || length + piece.length + 1 > size)) {
        length -= current[0].length + 1;
        current.shift();
      }
    }
    current.push(piece);
    length += piece.length + 1;
  }
  if (current.length) chunks.push(join(current));
  return chunks.filter((chunk) => chunk.length > 0);
}
