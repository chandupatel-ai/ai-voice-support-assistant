"""
AI Voice Support Assistant - Backend
=====================================
Real-time voice support pipeline:
  1. Real-time transcription (Groq Whisper large-v3-turbo)
  2. Agent response suggestions (Groq Llama 3.3 70B)
  3. Customer sentiment detection (Groq Llama 3.3 70B)
  4. Post-call summary + action items (Groq Llama 3.3 70B)

Stack: FastAPI, WebSockets, Groq API.
Pydantic v1 compatible.
"""

import os
import json
import time
import uuid
import tempfile
from typing import List, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
TRANSCRIBE_MODEL = "whisper-large-v3-turbo"
LLM_MODEL = "llama-3.3-70b-versatile"

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

app = FastAPI(title="AI Voice Support Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory call store
CALLS: Dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Models (pydantic v1 compatible)
# ---------------------------------------------------------------------------

class SummaryRequest(BaseModel):
    call_id: str


class SentimentResult(BaseModel):
    label: str
    score: float
    reason: str


# ---------------------------------------------------------------------------
# Helpers - Groq calls
# ---------------------------------------------------------------------------

def transcribe_audio_bytes(audio_bytes: bytes, filename: str = "chunk.wav") -> str:
    if not client:
        raise RuntimeError("GROQ_API_KEY not configured on server")

    suffix = os.path.splitext(filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            result = client.audio.transcriptions.create(
                file=(filename, f.read()),
                model=TRANSCRIBE_MODEL,
                response_format="text",
            )
    finally:
        os.unlink(tmp_path)

    return str(result).strip()


def analyze_sentiment(transcript: str) -> SentimentResult:
    if not transcript.strip():
        return SentimentResult(label="neutral", score=0.5, reason="No speech yet")

    prompt = f"""You are a call-center sentiment engine. Analyze the CUSTOMER's sentiment
in this support call transcript so far. Respond ONLY with JSON:
{{"label": "positive|neutral|negative|frustrated|angry", "score": 0.0-1.0, "reason": "short reason"}}

Transcript:
{transcript}
"""
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=200,
    )
    raw = resp.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(raw)
        return SentimentResult(**data)
    except Exception:
        return SentimentResult(label="neutral", score=0.5, reason="parse_error")


def suggest_agent_response(transcript: str) -> List[str]:
    if not transcript.strip():
        return []

    prompt = f"""You are an AI copilot for a customer support agent, listening live to a call.
Based on the transcript so far, suggest up to 3 short, ready-to-say responses the agent
could use next. Respond ONLY with a JSON array of strings, no preamble.

Transcript:
{transcript}
"""
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=250,
    )
    raw = resp.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except Exception:
        return [raw]


def generate_summary(transcript: str) -> dict:
    prompt = f"""Summarize this customer support call transcript. Respond ONLY with JSON:
{{
  "summary": "2-3 sentence summary",
  "issue": "the core customer issue",
  "resolution": "what was resolved or agreed, if anything",
  "action_items": ["action 1", "action 2"],
  "overall_sentiment": "positive|neutral|negative"
}}

Transcript:
{transcript}
"""
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=400,
    )
    raw = resp.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except Exception:
        return {"summary": raw, "issue": "", "resolution": "", "action_items": [], "overall_sentiment": "neutral"}


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def health():
    return {"status": "ok", "service": "ai-voice-support-assistant"}


@app.post("/calls/start")
def start_call():
    call_id = str(uuid.uuid4())
    CALLS[call_id] = {"transcript": "", "started_at": time.time(), "turns": []}
    return {"call_id": call_id}


@app.post("/calls/{call_id}/transcribe")
async def transcribe_chunk(call_id: str, file: UploadFile = File(...)):
    if call_id not in CALLS:
        raise HTTPException(404, "call not found")
    audio_bytes = await file.read()
    text = transcribe_audio_bytes(audio_bytes, file.filename or "chunk.wav")
    if text:
        CALLS[call_id]["transcript"] += f" {text}"
        CALLS[call_id]["turns"].append({"t": time.time(), "text": text})
    transcript = CALLS[call_id]["transcript"]
    sentiment = analyze_sentiment(transcript)
    suggestions = suggest_agent_response(transcript)
    return {
        "text": text,
        "transcript": transcript,
        "sentiment": sentiment.dict(),
        "suggestions": suggestions,
    }


@app.post("/calls/{call_id}/summary")
def call_summary(call_id: str):
    if call_id not in CALLS:
        raise HTTPException(404, "call not found")
    transcript = CALLS[call_id]["transcript"]
    return generate_summary(transcript)


# ---------------------------------------------------------------------------
# WebSocket endpoint - real-time streaming pipeline
# ---------------------------------------------------------------------------

@app.websocket("/ws/{call_id}")
async def call_stream(websocket: WebSocket, call_id: str):
    await websocket.accept()
    CALLS.setdefault(call_id, {"transcript": "", "started_at": time.time(), "turns": []})

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"] is not None:
                audio_bytes = message["bytes"]
                text = transcribe_audio_bytes(audio_bytes, "chunk.webm")
                if text:
                    CALLS[call_id]["transcript"] += f" {text}"
                transcript = CALLS[call_id]["transcript"]
                sentiment = analyze_sentiment(transcript)
                suggestions = suggest_agent_response(transcript)
                await websocket.send_json({
                    "type": "update",
                    "text": text,
                    "transcript": transcript,
                    "sentiment": sentiment.dict(),
                    "suggestions": suggestions,
                })

            elif "text" in message and message["text"] is not None:
                payload = json.loads(message["text"])
                if payload.get("type") == "end":
                    summary = generate_summary(CALLS[call_id]["transcript"])
                    await websocket.send_json({"type": "summary", "data": summary})

    except WebSocketDisconnect:
        pass