# AI Voice Support Assistant

Real-time voice support assistant: live transcription, agent response suggestions,
customer sentiment detection, and post-call summaries with action items.

**Stack:** FastAPI + WebSockets (backend), Groq Whisper large-v3-turbo (transcription),
Groq Llama 3.3 70B (suggestions/sentiment/summary), vanilla JS + MediaRecorder/WebRTC (frontend).

## How it works

1. Frontend captures mic audio via `getUserMedia` (WebRTC) and streams ~4s chunks over a WebSocket.
2. Backend transcribes each chunk with Groq Whisper, appends it to the running transcript.
3. On every chunk, backend also runs sentiment analysis and generates suggested agent
   responses from the running transcript, and pushes both back to the frontend live.
4. When the agent clicks "End Call", backend generates a structured summary (issue,
   resolution, action items, overall sentiment).

## 1. Backend — deploy to Render

```bash
cd backend
# locally:
pip install -r requirements.txt
export GROQ_API_KEY=your_key
uvicorn main:app --reload
```

**Deploy on Render (free tier is fine for a demo):**
1. Push this repo to GitHub.
2. On render.com → New → Web Service → connect the repo, root directory `backend`.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add environment variable `GROQ_API_KEY` (get a free key at console.groq.com).
6. Deploy. Note the URL, e.g. `https://ai-voice-support.onrender.com` — the
   WebSocket URL will be `wss://ai-voice-support.onrender.com`.

## 2. Frontend — deploy to Vercel

The frontend is a single static `index.html` — no build step needed.

1. `cd frontend`
2. `vercel --prod` (or drag-and-drop the `frontend` folder into vercel.com/new)
3. Open the deployed URL, paste your backend's `wss://...` URL into the
   "Backend WebSocket URL" field, and click **Start Call**.

## 3. Get a Groq API key

Free at https://console.groq.com/keys — used for both Whisper transcription and
the Llama 3.3 70B chat completions.

## Evaluation criteria mapping

| Criteria | Where it's addressed |
|---|---|
| Transcription latency | 4s chunked streaming over WebSocket + Groq's low-latency Whisper turbo model |
| Suggestion relevance | Prompted on the full running transcript, not just the last chunk |
| Sentiment accuracy | Structured JSON classification (positive/neutral/negative/frustrated/angry) with a confidence score |
| Summary quality | Structured summary: issue, resolution, action items, overall sentiment |
| System reliability | Stateless REST fallback endpoints (`/calls/{id}/transcribe`, `/calls/{id}/summary`) in addition to the WebSocket path; JSON parse fallbacks on every LLM call |

## Project structure

```
backend/
  main.py            FastAPI app: REST + WebSocket endpoints, Groq integration
  requirements.txt
  .env.example
frontend/
  index.html          Single-file UI: mic capture, live transcript, sentiment, suggestions, summary
README.md
```
