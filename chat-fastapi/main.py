import base64
import json
import logging
import re
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import settings
from utils import (
    Message,
    client,
    generate_image,
    moderate_text,
    speech_to_text,
    speech_to_translation,
    text_to_speech,
    tmp_audio_file,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Chatbot App")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Generic moderation error ──────────────────────────────────────────────────
MODERATION_ERROR = "🚫 Ce contenu ne peut pas être traité."


def _check_moderation(text: str) -> None:
    """
    Run moderation on `text`. Raises HTTP 422 with a generic message if flagged.
    Flagged categories are logged server-side only — never exposed to the client.
    """
    if not text.strip():
        return
    result = moderate_text(text)
    if result.flagged:
        logger.warning("Content blocked by moderation: %s", result.flagged_categories)
        raise HTTPException(status_code=422, detail=MODERATION_ERROR)


# ── Pydantic models ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    messages: list[Message]
    model: str = settings.default_text_model
    temperature: float = Field(settings.default_temperature, ge=0.0, le=2.0)


class ImageRequest(BaseModel):
    prompt: str
    model: str = settings.default_image_model
    size: str = settings.default_image_size


class TTSRequest(BaseModel):
    text: str = Field(..., max_length=4096)
    model: str = settings.default_tts_model
    voice: str = settings.default_tts_voice
    speed: float = Field(settings.default_tts_speed, ge=0.5, le=2.0)


class ModerateRequest(BaseModel):
    text: str


# ── SSE helper ────────────────────────────────────────────────────────────────

def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def index() -> FileResponse:
    return FileResponse("static/index.html")


# TEXT CHAT — streaming SSE
@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    last_user = next((m["content"] for m in reversed(req.messages) if m["role"] == "user"), "")
    _check_moderation(last_user)

    full_messages = [{"role": "system", "content": settings.system_message}] + req.messages

    def generate():
        try:
            stream = client.chat.completions.create(
                model=req.model,
                messages=full_messages,
                temperature=req.temperature,
                stream=True,
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                if token:
                    yield sse({"token": token})
            yield sse({"done": True})
        except Exception as exc:
            logger.exception("Chat stream error")
            yield sse({"error": str(exc)})

    return StreamingResponse(generate(), media_type="text/event-stream")


# IMAGE
@app.post("/api/image")
def image(req: ImageRequest) -> dict:
    _check_moderation(req.prompt)
    try:
        image_bytes = generate_image(req.prompt, model=req.model, size=req.size)
        return {"image_b64": base64.b64encode(image_bytes).decode()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# SPEECH TO TEXT
@app.post("/api/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    model: str = Form(settings.default_stt_model),
) -> dict:
    with tmp_audio_file(".webm") as tmp_path:
        tmp_path.write_bytes(await audio.read())
        try:
            text = speech_to_text(tmp_path, model=model)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    _check_moderation(text)
    return {"text": text}


# SPEECH TO TRANSLATION
@app.post("/api/translate")
async def translate(audio: UploadFile = File(...)) -> dict:
    with tmp_audio_file(".webm") as tmp_path:
        tmp_path.write_bytes(await audio.read())
        try:
            text = speech_to_translation(tmp_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    _check_moderation(text)
    return {"text": text}


# TEXT TO SPEECH
@app.post("/api/tts")
def tts(req: TTSRequest) -> FileResponse:
    _check_moderation(req.text)
    out_path = settings.tmp_dir / f"{uuid.uuid4()}.wav"
    try:
        text_to_speech(req.text, output_path=out_path,
                       model=req.model, voice=req.voice, speed=req.speed)
        return FileResponse(str(out_path), media_type="audio/wav")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# VOICE CHAT — streaming SSE (STT → modération → LLM stream → TTS per sentence)
_SENTENCE_RE = re.compile(r'(?<=[.!?…])\s+')


def _split_sentences(text: str) -> list[str]:
    """Split text on sentence boundaries, keeping punctuation attached."""
    return [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]


def _tts_to_b64(text: str, model: str, voice: str, speed: float) -> str:
    """Synthesise text and return base64-encoded WAV."""
    out_path = settings.tmp_dir / f"{uuid.uuid4()}.wav"
    try:
        text_to_speech(text, output_path=out_path, model=model, voice=voice, speed=speed)
        return base64.b64encode(out_path.read_bytes()).decode()
    finally:
        out_path.unlink(missing_ok=True)


@app.post("/api/voice-chat/stream")
async def voice_chat_stream(
    audio: UploadFile = File(...),
    history: str = Form("[]"),
    text_model: str = Form(settings.default_text_model),
    temperature: float = Form(settings.default_temperature),
    stt_model: str = Form(settings.default_stt_model),
    tts_model: str = Form(settings.default_tts_model),
    voice: str = Form(settings.default_tts_voice),
    speed: float = Form(settings.default_tts_speed),
) -> StreamingResponse:

    # 1. Transcribe
    with tmp_audio_file(".webm") as tmp_path:
        tmp_path.write_bytes(await audio.read())
        try:
            user_text = speech_to_text(tmp_path, model=stt_model)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # 2. Moderate transcribed text
    _check_moderation(user_text)

    # 3. Build conversation
    messages: list[Message] = json.loads(history)
    messages.append({"role": "user", "content": user_text})
    full_messages = [{"role": "system", "content": settings.system_message}] + messages

    def generate():
        yield sse({"user_text": user_text})

        buffer = ""
        full_response = ""

        try:
            stream = client.chat.completions.create(
                model=text_model,
                messages=full_messages,
                temperature=temperature,
                stream=True,
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                if not token:
                    continue

                buffer += token
                full_response += token
                yield sse({"token": token})

                sentences = _split_sentences(buffer)
                if len(sentences) > 1:
                    to_speak = " ".join(sentences[:-1])
                    buffer = sentences[-1]
                    audio_b64 = _tts_to_b64(to_speak, tts_model, voice, speed)
                    yield sse({"audio": audio_b64})

            if buffer.strip():
                audio_b64 = _tts_to_b64(buffer.strip(), tts_model, voice, speed)
                yield sse({"audio": audio_b64})

            yield sse({"done": True, "full_response": full_response})

        except Exception as exc:
            logger.exception("Voice chat stream error")
            yield sse({"error": str(exc)})

    return StreamingResponse(generate(), media_type="text/event-stream")


# MODERATION — standalone endpoint
@app.post("/api/moderate")
def moderate(req: ModerateRequest) -> dict:
    """Returns only flagged status — scores/categories stay server-side."""
    try:
        result = moderate_text(req.text)
        return {"flagged": result.flagged}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc