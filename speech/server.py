import os
import base64
import struct
import logging
import time
import httpx
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google import genai
from google.genai import types

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("AvatarPipeline")

load_dotenv()

app = FastAPI(title="Avatar Pipeline API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ────────────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DID_API_KEY    = os.getenv("DID_API_KEY")

# Default avatar image (D-ID stock photo)
DEFAULT_AVATAR_URL = os.getenv(
    "AVATAR_URL",
    "https://d-id-public-bucket.s3.us-west-2.amazonaws.com/alice.jpg"
)

VALID_VOICES = ["Puck", "Charon", "Kore", "Fenrir", "Zephyr"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_genai_client() -> genai.Client:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not set.")
    return genai.Client(api_key=GEMINI_API_KEY)


def get_did_headers() -> dict:
    """D-ID uses HTTP Basic auth: base64(api_key:)"""
    if not DID_API_KEY:
        raise HTTPException(status_code=500, detail="DID_API_KEY is not set.")
    token = base64.b64encode(f"{DID_API_KEY}:".encode()).decode()
    return {
        "Authorization": f"Basic {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 24000, channels: int = 1, bit_depth: int = 16) -> bytes:
    """Wrap raw PCM bytes in a WAV container."""
    byte_rate    = sample_rate * channels * bit_depth // 8
    block_align  = channels * bit_depth // 8
    data_size    = len(pcm_bytes)
    chunk_size   = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", chunk_size, b"WAVE",
        b"fmt ", 16,
        1,            # PCM
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bit_depth,
        b"data", data_size,
    )
    return header + pcm_bytes


async def create_did_talk(wav_bytes: bytes, avatar_url: str) -> dict:
    headers_auth = {
        "Authorization": f"Basic {base64.b64encode(f'{DID_API_KEY}:'.encode()).decode()}",
        "Accept": "application/json",
    }

    async with httpx.AsyncClient(timeout=40) as client:
        # 1. Upload audio to D-ID
        resp = await client.post(
            "https://api.d-id.com/audios",
            headers=headers_auth,
            files={"audio": ("speech.wav", wav_bytes, "audio/wav")},
        )
        if resp.status_code not in (200, 201):
            raise HTTPException(status_code=resp.status_code,
                                detail=f"D-ID audio upload failed: {resp.text}")

        audio_url = resp.json().get("url")
        logger.info(f"D-ID audio uploaded: {audio_url}")

        # 2. Create talk with audio URL
        payload = {
            "source_url": avatar_url,
            "script": {
                "type": "audio",
                "audio_url": audio_url,
            },
        }
        resp = await client.post(
            "https://api.d-id.com/talks",
            json=payload,
            headers={**headers_auth, "Content-Type": "application/json"},
        )
        if resp.status_code not in (200, 201):
            raise HTTPException(status_code=resp.status_code,
                                detail=f"D-ID create talk failed: {resp.text}")

        talk_id = resp.json().get("id")
        logger.info(f"D-ID talk created: {talk_id}")

        # 3. Poll until done
        for _ in range(30):
            time.sleep(1)
            poll = await client.get(
                f"https://api.d-id.com/talks/{talk_id}",
                headers=headers_auth,
            )
            data = poll.json()
            status = data.get("status")
            logger.info(f"D-ID talk status: {status}")
            if status == "done":
                return data
            if status == "error":
                raise HTTPException(status_code=500,
                                    detail=f"D-ID talk error: {data.get('error')}")

    raise HTTPException(status_code=504, detail="D-ID talk timed out.")

# ── Request Models ────────────────────────────────────────────────────────────

class ChatTurn(BaseModel):
    role: str
    text: str


class AvatarChatRequest(BaseModel):
    audioBase64:   Optional[str]        = None   # base64 webm/wav from browser mic
    audioMimeType: Optional[str]        = "audio/webm"
    inputText:     Optional[str]        = ""     # fallback text input
    voice:         Optional[str]        = "Kore"
    chatHistory:   Optional[List[ChatTurn]] = []
    avatarUrl:     Optional[str]        = None   # override default avatar image


class TtsOnlyRequest(BaseModel):
    text:  str
    voice: Optional[str] = "Kore"
    mood:  Optional[str] = "naturally"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/api/avatar-chat")
async def avatar_chat(payload: AvatarChatRequest):
    """
    Full pipeline:
      Mic audio (base64) → Gemini STT → Gemini LLM → Gemini TTS → D-ID Talk → video URL
    """
    client = get_genai_client()
    avatar_url = payload.avatarUrl or DEFAULT_AVATAR_URL

    # ── Step 1: STT ──────────────────────────────────────────────────────────
    spoken_query = ""
    if payload.audioBase64:
        audio_bytes = base64.b64decode(payload.audioBase64)
        stt_resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type=payload.audioMimeType or "audio/webm"),
                "Transcribe the speech. If inaudible return '[Inaudible]'."
            ],
        )
        spoken_query = (stt_resp.text or "").strip()
    else:
        spoken_query = (payload.inputText or "").strip()

    if not spoken_query or spoken_query == "[Inaudible]":
        return {"userMessage": spoken_query or "[No speech]", "aiMessage": None, "videoUrl": None}

    logger.info(f"STT result: {spoken_query}")

    # ── Step 2: LLM ──────────────────────────────────────────────────────────
    directive = (
        "You are a concise, friendly voice assistant. "
        "Keep replies to 1-3 sentences."
    )
    contents = []
    for turn in (payload.chatHistory or []):
        role = "user" if turn.role == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=turn.text)]))
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=spoken_query)]))

    llm_resp = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=directive, temperature=0.8),
    )
    ai_text = (llm_resp.text or "").strip()
    logger.info(f"LLM reply: {ai_text}")

    # ── Step 3: TTS ──────────────────────────────────────────────────────────
    voice_name = payload.voice if payload.voice in VALID_VOICES else "Kore"
    tts_resp = client.models.generate_content(
        model="gemini-3.1-flash-tts-preview",
        contents=f"Say naturally: {ai_text}",
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                )
            ),
        ),
    )

    pcm_raw = None
    for candidate in (tts_resp.candidates or []):
        for part in (candidate.content.parts or []):
            if part.inline_data and part.inline_data.data:
                pcm_raw = part.inline_data.data
                break

    if not pcm_raw:
        raise HTTPException(status_code=500, detail="Gemini TTS returned no audio.")

    if isinstance(pcm_raw, str):
        pcm_raw = base64.b64decode(pcm_raw)

    # Convert PCM → WAV, then re-encode as base64 for D-ID
    wav_bytes   = pcm_to_wav(pcm_raw, sample_rate=24000)
    wav_b64     = base64.b64encode(wav_bytes).decode()
    logger.info("TTS audio ready, sending to D-ID...")

    # ── Step 4: D-ID Talk API ────────────────────────────────────────────────
    talk_data = await create_did_talk(wav_bytes, avatar_url)
    video_url = talk_data.get("result_url")

    return {
        "userMessage": spoken_query,
        "aiMessage":   ai_text,
        "videoUrl":    video_url,
        "talkId":      talk_data.get("id"),
    }


@app.post("/api/tts")
async def tts_only(payload: TtsOnlyRequest):
    """TTS only — returns base64 WAV audio."""
    client = get_genai_client()
    voice_name = payload.voice if payload.voice in VALID_VOICES else "Kore"

    resp = client.models.generate_content(
        model="gemini-3.1-flash-tts-preview",
        contents=f"Say {payload.mood}: {payload.text}",
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                )
            ),
        ),
    )

    pcm_raw = None
    for candidate in (resp.candidates or []):
        for part in (candidate.content.parts or []):
            if part.inline_data and part.inline_data.data:
                pcm_raw = part.inline_data.data
                break

    if not pcm_raw:
        raise HTTPException(status_code=500, detail="No audio from Gemini TTS.")

    if isinstance(pcm_raw, str):
        pcm_raw = base64.b64decode(pcm_raw)

    wav_bytes = pcm_to_wav(pcm_raw)
    return {
        "audioBase64": base64.b64encode(wav_bytes).decode(),
        "sampleRate": 24000,
        "text": payload.text,
        "voice": voice_name,
    }


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "gemini_key": bool(GEMINI_API_KEY),
        "did_key":    bool(DID_API_KEY),
        "avatar_url": DEFAULT_AVATAR_URL,
    }


if os.path.exists("index.html"):
    @app.get("/")
    async def index():
        return FileResponse("index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=3000, reload=True)