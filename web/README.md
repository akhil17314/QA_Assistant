# Document Q&A: Vercel web app

Stateless Next.js port of the Streamlit app. Gemini handles the LLM and embeddings, and the browser holds each session's index (nothing is stored on the server).

## Deploy to Vercel
1. Push the repo to GitHub.
2. In Vercel, choose **Add New → Project**, import `QA_Assistant`, and set **Root Directory** to `web`.
3. Add the environment variable `GOOGLE_API_KEY`. `LLM_MODEL`, `JUDGE_MODEL` and `EMBEDDING_MODEL` are optional.
4. Deploy.

CLI alternative: `cd web; npx vercel --prod`, then set the root directory when prompted.

## Local
```
cd web
$env:GOOGLE_API_KEY="..."
npm run dev
```

## Limits
- PDFs of 4 MB or less each (Vercel's request body limit).
- Reloading the tab clears the index, as with the Streamlit app.

## Errors
Upload and chat errors appear in the app. Gemini errors include the relevant HTTP status and
suggested next steps; server logs contain the full diagnostic details. Unexpected internal
errors are not displayed verbatim to avoid exposing server configuration or secrets.
