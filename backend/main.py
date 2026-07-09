import os
import json
import time
import uuid
import tempfile
from typing import List, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
TRANSCRIBE_MODEL = "whisper-large-v3-turbo"
LLM_MODEL = "llama-3.3-70b-versatile"

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

app = FastAPI(title="AI Voice Support Assistant")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

CALLS: Dict[str, dict] = {}

def transcribe_audio_bytes(audio_bytes: bytes, filename: str = "chunk.wav") -> str:
    if not client:
        raise RuntimeError("GROQ_API_KEY not configured")
    suffix = os.path.splitext(filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            result = client.audio.transcriptions.create(file=(filename, f.read()), model=TRANSCRIBE_MODEL, response_format="text")
    finally:
        os.unlink(tmp_path)
    return str(result).strip()

def analyze_sentiment(transcript: str) -> dict:
    if not transcript.strip():
        return {"label": "neutral", "score": 0.5, "reason": "No speech yet"}
    prompt = f'You are a call-center sentiment engine. Respond ONLY with JSON: {{"label": "positive|neutral|negative|frustrated|angry", "score": 0.0-1.0, "reason": "short reason"}}\nTranscript: {transcript}'
    resp = client.chat.completions.create(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.2, max_tokens=200)
    raw = resp.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
    try:
        return json.loads(raw)
    except:
        return {"label": "neutral", "score": 0.5, "reason": "parse_error"}

def suggest_agent_response(transcript: str) -> List[str]:
    if not transcript.strip():
        return []
    prompt = f'You are an AI copilot for a support agent. Suggest up to 3 short responses. Respond ONLY with a JSON array of strings.\nTranscript: {transcript}'
    resp = client.chat.completions.create(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.4, max_tokens=250)
    raw = resp.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
    try:
        return json.loads(raw)
    except:
        return [raw]

def generate_summary(transcript: str) -> dict:
    prompt = f'Summarize this support call. Respond ONLY with JSON: {{"summary": "2-3 sentences", "issue": "core issue", "resolution": "what was resolved", "action_items": ["item1"], "overall_sentiment": "positive|neutral|negative"}}\nTranscript: {transcript}'
    resp = client.chat.completions.create(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.2, max_tokens=400)
    raw = resp.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
    try:
        return json.loads(raw)
    except:
        return {"summary": raw, "issue": "", "resolution": "", "action_items": [], "overall_sentiment": "neutral"}

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
    return {"text": text, "transcript": transcript, "sentiment": analyze_sentiment(transcript), "suggestions": suggest_agent_response(transcript)}

@app.post("/calls/{call_id}/summary")
def call_summary(call_id: str):
    if call_id not in CALLS:
        raise HTTPException(404, "call not found")
    return generate_summary(CALLS[call_id]["transcript"])

@app.websocket("/ws/{call_id}")
async def call_stream(websocket: WebSocket, call_id: str):
    await websocket.accept()
    CALLS.setdefault(call_id, {"transcript": "", "started_at": time.time(), "turns": []})
    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"] is not None:
                text = transcribe_audio_bytes(message["bytes"], "chunk.webm")
                if text:
                    CALLS[call_id]["transcript"] += f" {text}"
                transcript = CALLS[call_id]["transcript"]
                await websocket.send_json({"type": "update", "text": text, "transcript": transcript, "sentiment": analyze_sentiment(transcript), "suggestions": suggest_agent_response(transcript)})
            elif "text" in message and message["text"] is not None:
                payload = json.loads(message["text"])
                if payload.get("type") == "end":
                    await websocket.send_json({"type": "summary", "data": generate_summary(CALLS[call_id]["transcript"])})
    except WebSocketDisconnect:
        pass
