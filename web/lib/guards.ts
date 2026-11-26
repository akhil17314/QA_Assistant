export const MAX_QUERY_LENGTH = 2000;

const INJECTION_PATTERNS = [
  /\bignore\s+(all\s+)?(previous|prior|above)\s+instructions?\b/i,
  /\b(reveal|show|print|repeat|expose)\s+(your\s+)?(system|developer)\s+prompt\b/i,
  /\byou\s+are\s+now\b/i,
  /\bdeveloper\s+mode\b/i,
  /\bact\s+as\s+(dan|an?\s+unrestricted)\b/i,
  /\bextract\s+(the\s+)?(api\s+key|secrets?|credentials?)\b/i,
];

const API_KEY_PATTERN = /\b(?:AIza[0-9A-Za-z_-]{30,}|sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,})\b/g;
const EMAIL_PATTERN = /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi;

export function sanitizeInput(text: string): string {
  // eslint-disable-next-line no-control-regex
  return text.replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/g, "");
}

export function validateInputRules(text: string): { blocked: boolean; reason: string } {
  if (text.length > MAX_QUERY_LENGTH) {
    return { blocked: true, reason: `Message exceeds the ${MAX_QUERY_LENGTH}-character limit.` };
  }
  if (INJECTION_PATTERNS.some((pattern) => pattern.test(text))) {
    return { blocked: true, reason: "The message matches a prompt-injection or secret-extraction pattern." };
  }
  return { blocked: false, reason: "" };
}

export function redactSecrets(text: string): string {
  return text.replace(API_KEY_PATTERN, "[REDACTED KEY]").replace(EMAIL_PATTERN, "[REDACTED EMAIL]");
}
