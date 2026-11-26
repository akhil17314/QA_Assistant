"use client";

import { FormEvent, useRef, useState } from "react";

const TOP_K = 6;
const TOP_K_RETRY = 10;
const MAX_RETRIES = 1;
const MAX_UPLOAD_BYTES = 4 * 1024 * 1024;

type Chunk = { text: string; source: string; page: number; embedding: number[] };
type Citation = { source: string; page: number; snippet: string };
type Message = {
  role: "user" | "assistant";
  content: string;
  error?: boolean;
  refused?: boolean;
  citations?: Citation[];
  route?: string;
  retries?: number;
  grounded?: boolean;
  blockReason?: string;
};

// Embeddings are L2-normalized server-side, so a dot product equals cosine similarity.
function rank(query: number[], chunks: Chunk[], k: number) {
  return chunks
    .map((chunk) => ({ chunk, score: chunk.embedding.reduce((sum, v, i) => sum + v * query[i], 0) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k)
    .map(({ chunk }) => ({ text: chunk.text, source: chunk.source, page: chunk.page }));
}

async function post<T>(url: string, body: BodyInit, json = true): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, {
      method: "POST",
      body,
      headers: json ? { "Content-Type": "application/json" } : undefined,
    });
  } catch (error) {
    console.error("Network request failed", error);
    throw new Error("Could not reach the server. Check your connection and try again.");
  }
  let data: unknown;
  try {
    data = await response.json();
  } catch (error) {
    console.error("Invalid server response", error);
    throw new Error(
      response.ok
        ? "The server returned an invalid response. Please try again."
        : `Request failed (HTTP ${response.status}). The server did not return error details.`,
    );
  }
  if (!response.ok) {
    const message = typeof data === "object" && data !== null && "error" in data && typeof data.error === "string"
      ? data.error
      : `Request failed (HTTP ${response.status}).`;
    throw new Error(message);
  }
  return data as T;
}

export default function Home() {
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [notices, setNotices] = useState<{ text: string; error: boolean }[]>([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [ingesting, setIngesting] = useState(false);
  const [selectedFileCount, setSelectedFileCount] = useState(0);
  const fileInput = useRef<HTMLInputElement>(null);
  const lastAsked = useRef(0);

  const sources = [...new Set(chunks.map((chunk) => chunk.source))].sort();
  const chunkCount = chunks.length;

  async function ingest() {
    const files = Array.from(fileInput.current?.files ?? []);
    if (!files.length) return setNotices([{ text: "Choose at least one PDF first.", error: true }]);
    setIngesting(true);
    setNotices([]);
    for (const file of files) {
      try {
        if (file.size > MAX_UPLOAD_BYTES) throw new Error("File exceeds the 4 MB upload limit.");
        const form = new FormData();
        form.append("file", file);
        const result = await post<{ source: string; chunks: Chunk[] }>("/api/ingest", form, false);
        setChunks((current) => [...current.filter((c) => c.source !== result.source), ...result.chunks]);
        setNotices((current) => [...current, { text: `Indexed ${file.name}: ${result.chunks.length} chunks`, error: false }]);
      } catch (error) {
        const message = error instanceof Error ? error.message : "An unexpected error occurred.";
        setNotices((current) => [...current, { text: `Could not ingest ${file.name}: ${message}`, error: true }]);
      }
    }
    if (fileInput.current) fileInput.current.value = "";
    setSelectedFileCount(0);
    setIngesting(false);
  }

  function clearIndex() {
    setChunks([]);
    setMessages([]);
    setNotices([]);
  }

  function history() {
    return messages
      .filter((message, index) => {
        const next = messages[index + 1];
        if (message.error || message.refused) return false;
        return !(message.role === "user" && next && (next.error || next.refused));
      })
      .map(({ role, content }) => ({ role, content }));
  }

  async function ask(event: FormEvent) {
    event.preventDefault();
    const query = question.trim();
    if (!query || busy) return;
    if (Date.now() - lastAsked.current < 3000) {
      return setNotices([{ text: "Please wait a few seconds between questions.", error: true }]);
    }
    lastAsked.current = Date.now();
    const prior = history();
    setMessages((current) => [...current, { role: "user", content: query }]);
    setQuestion("");
    setBusy(true);
    const reply = (message: Omit<Message, "role">) =>
      setMessages((current) => [...current, { role: "assistant", ...message }]);

    try {
      if (!chunks.length) {
        return reply({ content: "Please upload and ingest a PDF before asking a document question.", route: "off_topic" });
      }
      const analysis = await post<{
        blocked: boolean;
        reason?: string;
        route?: string;
        message?: string;
        query?: string;
        embedding?: number[];
      }>("/api/analyze", JSON.stringify({ query, history: prior }));
      if (analysis.blocked) {
        return reply({ content: analysis.message ?? "", refused: true, blockReason: analysis.reason, route: "blocked" });
      }
      if (analysis.route === "off_topic" || !analysis.embedding) {
        return reply({ content: analysis.message ?? "", route: "off_topic" });
      }

      let retries = 0;
      let k = TOP_K;
      for (;;) {
        const result = await post<{
          answer: string;
          citations: Citation[];
          grounded: boolean;
        }>(
          "/api/answer",
          JSON.stringify({ query: analysis.query, history: prior, chunks: rank(analysis.embedding, chunks, k) }),
        );
        if (result.grounded) {
          return reply({ content: result.answer, citations: result.citations, route: "in_scope", retries, grounded: true });
        }
        if (retries >= MAX_RETRIES) {
          return reply({
            content: "I could not find a well-supported answer in the uploaded documents.",
            route: "in_scope",
            retries,
            grounded: false,
          });
        }
        retries += 1;
        k = TOP_K_RETRY;
      }
    } catch (error) {
      reply({ content: error instanceof Error ? error.message : "An unexpected error occurred.", error: true });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="workspace">
      <aside className="sidebar">
        <a className="brand" href="/" aria-label="Document Q&A home">
          <span className="brand-mark">D</span>
          <span className="brand-name">folio<span className="brand-period">.</span></span>
        </a>

        <div className="sidebar-section">
          <div className="section-heading">YOUR LIBRARY</div>
          <div className="library-card">
            <span className="library-icon" aria-hidden="true">▤</span>
            <span className="library-copy">
              <strong>My documents</strong>
              <small>{sources.length} {sources.length === 1 ? "file" : "files"} · {chunkCount} passages</small>
            </span>
            <span className="library-dot" aria-hidden="true" />
          </div>
        </div>

        <div className="sidebar-section document-section">
          <div className="section-heading section-heading-row">
            <span>DOCUMENTS</span>
            {sources.length > 0 && <span className="count-pill">{sources.length}</span>}
          </div>
          {sources.length ? (
            <ul className="source-list">
              {sources.map((source) => (
                <li key={source}>
                  <span className="pdf-icon" aria-hidden="true">PDF</span>
                  <span title={source}>{source}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="sidebar-empty">Your library is waiting for its first document.</p>
          )}
        </div>

        <div className="sidebar-bottom">
          <div className="privacy-note">
            <span className="privacy-icon" aria-hidden="true">◈</span>
            <span><strong>Private by design</strong><small>Files stay in this browser tab.</small></span>
          </div>
          <button className="clear-button" onClick={clearIndex} disabled={!chunks.length && !messages.length}>
            Clear workspace
          </button>
        </div>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <div className="breadcrumb"><span>Workspace</span><span className="breadcrumb-slash">/</span><strong>Document Q&amp;A</strong></div>
          <div className="topbar-status"><span className="status-dot" /> Session only</div>
        </header>

        <section className="conversation">
          <div className="intro">
            <div className="intro-mark" aria-hidden="true">✳</div>
            <p className="eyebrow">YOUR READING ROOM</p>
            <h1>Good questions<br /><em>start here.</em></h1>
            <p className="intro-copy">
              Bring your PDFs together. Ask what matters. Get answers that point back to the page.
            </p>
            <label className="upload-card">
              <input
                ref={fileInput}
                type="file"
                accept="application/pdf"
                multiple
                onChange={(event) => setSelectedFileCount(event.target.files?.length ?? 0)}
              />
              <span className="upload-symbol" aria-hidden="true">↑</span>
              <span className="upload-text">
                <strong>
                  {selectedFileCount
                    ? `${selectedFileCount} ${selectedFileCount === 1 ? "PDF" : "PDFs"} selected`
                    : sources.length ? "Add more documents" : "Choose PDFs to get started"}
                </strong>
                <small>Up to 4 MB each · PDF format</small>
              </span>
              <span className="upload-arrow" aria-hidden="true">↗</span>
            </label>
          </div>

          {notices.length > 0 && (
            <div className="notice-list" aria-live="polite">
              {notices.map((notice, index) => (
                <div key={index} className={`notice ${notice.error ? "error" : ""}`}>
                  <span aria-hidden="true">{notice.error ? "!" : "✓"}</span>{notice.text}
                </div>
              ))}
            </div>
          )}

          {selectedFileCount > 0 ? (
            <button className="ingest-button" onClick={ingest} disabled={ingesting}>
              {ingesting ? "Reading your documents…" : `Add ${selectedFileCount} ${selectedFileCount === 1 ? "document" : "documents"} to library`}
              {!ingesting && <span aria-hidden="true">→</span>}
            </button>
          ) : null}

          {messages.length > 0 && (
            <div className="messages" aria-live="polite">
              {messages.map((message, index) => (
                <div key={index} className={`message-row ${message.role}`}>
                  {message.role === "assistant" && <span className="assistant-mark" aria-hidden="true">✳</span>}
                  <div className={`message-card ${message.error ? "error" : message.role}`}>
                    {message.role === "assistant" && <div className="message-label">FOLIO ANSWER</div>}
                    <div className="message-content">{message.content}</div>
                    {message.role === "assistant" && !message.error && (
                      <>
                        <details className="message-details">
                          <summary>Sources <span>{message.citations?.length ?? 0}</span></summary>
                          {message.citations?.length ? (
                            message.citations.map((citation, citationIndex) => (
                              <div className="citation" key={citationIndex}>
                                <div><strong>{citation.source}</strong><span>PAGE {citation.page}</span></div>
                                <blockquote>{citation.snippet}</blockquote>
                              </div>
                            ))
                          ) : <p className="detail-empty">No document citations were returned.</p>}
                        </details>
                        <details className="message-details trace-details">
                          <summary>Answer details <span>{message.grounded ? "GROUNDED" : message.route?.toUpperCase() ?? "DETAILS"}</span></summary>
                          <div className="trace-content">
                            <span>Route <strong>{message.route ?? "unknown"}</strong></span>
                            <span>Retries <strong>{message.retries ?? 0}</strong></span>
                            <span>Grounded <strong>{message.grounded ? "Yes" : "No"}</strong></span>
                            {message.blockReason && <span>Reason <strong>{message.blockReason}</strong></span>}
                          </div>
                        </details>
                      </>
                    )}
                  </div>
                </div>
              ))}
              {busy && <div className="thinking"><span className="thinking-dot" /> Reading across your documents…</div>}
            </div>
          )}
        </section>

        <footer className="composer-wrap">
          <form className="composer" onSubmit={ask}>
            <input
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder={sources.length ? "Ask a question about your documents…" : "Add a PDF to start a conversation"}
              maxLength={2000}
              disabled={!sources.length || busy}
              aria-label="Ask a question about your documents"
            />
            <button className="send-button" disabled={busy || !question.trim() || !sources.length} aria-label="Send question">
              {busy ? <span className="send-spinner" /> : <span aria-hidden="true">↑</span>}
            </button>
          </form>
          <p className="composer-caption">Answers are grounded in your files. Always check the cited page.</p>
        </footer>
      </main>
    </div>
  );
}
