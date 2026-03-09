import base64
import json
import logging
import re
import uuid
from pathlib import Path

from typing import Optional
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import settings
from tools import TOOL_DEFINITIONS, execute_tool
from utils import (
    Message,
    SUPPORTED_UPLOAD_TYPES,
    build_message_with_file,
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


# TEXT CHAT — streaming SSE with tool calling
@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest, request: Request) -> StreamingResponse:
    last_user = next((m["content"] for m in reversed(req.messages) if m["role"] == "user"), "")
    _check_moderation(last_user)

    full_messages = [{"role": "system", "content": settings.system_message}] + req.messages
    base_url = str(request.base_url).rstrip("/")  # e.g. "http://127.0.0.1:8000"

    def generate():
        try:
            messages = list(full_messages)

            while True:
                stream = client.chat.completions.create(
                    model=req.model,
                    messages=messages,
                    temperature=req.temperature,
                    tools=TOOL_DEFINITIONS,
                    tool_choice="auto",
                    stream=True,
                )

                # Accumulate the streamed response
                tool_calls_acc: dict[int, dict] = {}  # index → {id, name, arguments}
                text_acc = ""
                finish_reason = None

                for chunk in stream:
                    delta = chunk.choices[0].delta
                    finish_reason = chunk.choices[0].finish_reason or finish_reason

                    # Stream text tokens to client
                    if delta.content:
                        text_acc += delta.content
                        yield sse({"token": delta.content})

                    # Accumulate tool call chunks
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index
                            if idx not in tool_calls_acc:
                                tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
                            if tc.id:
                                tool_calls_acc[idx]["id"] = tc.id
                            if tc.function.name:
                                tool_calls_acc[idx]["name"] = tc.function.name
                            if tc.function.arguments:
                                tool_calls_acc[idx]["arguments"] += tc.function.arguments

                # No tool calls → done
                if finish_reason != "tool_calls" or not tool_calls_acc:
                    yield sse({"done": True})
                    break

                # Append assistant message with tool_calls
                messages.append({
                    "role": "assistant",
                    "content": text_acc or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": tc["arguments"]},
                        }
                        for tc in tool_calls_acc.values()
                    ],
                })

                # Execute each tool and append results
                for tc in tool_calls_acc.values():
                    logger.info("LLM called tool '%s'", tc["name"])
                    yield sse({"tool_call": tc["name"]})  # notify frontend
                    result = execute_tool(tc["name"], tc["arguments"])
                    result_data = json.loads(result)

                    # If a file was generated, notify frontend with download info
                    if tc["name"] == "generate_file" and result_data.get("success"):
                        abs_url = base_url + result_data["download_url"]
                        # Patch result so LLM cites the absolute URL in its text response
                        result_data["download_url"] = abs_url
                        result = json.dumps(result_data)
                        yield sse({"file": {
                            "filename": result_data["filename"],
                            "url": abs_url,
                            "description": result_data["description"],
                        }})

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })
                # Loop: let LLM generate its final response after tool results

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



# FILE UPLOAD — chat with an attached file
@app.post("/api/chat/upload/stream")
async def chat_upload_stream(
    file: UploadFile = File(...),
    text: str = Form(""),
    history: str = Form("[]"),
    model: str = Form(settings.default_text_model),
    temperature: float = Form(settings.default_temperature),
) -> StreamingResponse:
    """
    Receive a file + optional text message, build a multimodal or text message,
    and stream the LLM response via SSE.
    """
    mime_type = file.content_type or "application/octet-stream"
    if mime_type not in SUPPORTED_UPLOAD_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Type de fichier non supporté : {mime_type}. Formats acceptés : PDF, TXT, CSV, JPG, PNG."
        )

    file_bytes = await file.read()
    if len(file_bytes) > 20 * 1024 * 1024:  # 20 MB limit
        raise HTTPException(status_code=413, detail="Fichier trop volumineux (max 20 Mo).")

    # Moderate text part if provided
    if text.strip():
        _check_moderation(text)

    # Build the enriched message
    try:
        file_message = build_message_with_file(text, file_bytes, mime_type, file.filename or "fichier")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Impossible de lire le fichier : {exc}") from exc

    # Build full conversation
    messages: list[Message] = json.loads(history)
    messages.append(file_message)
    full_messages = [{"role": "system", "content": settings.system_message}] + messages

    def generate():
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=full_messages,
                temperature=temperature,
                stream=True,
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                if token:
                    yield sse({"token": token})
            yield sse({"done": True})
        except Exception as exc:
            logger.exception("Chat upload stream error")
            yield sse({"error": str(exc)})

    return StreamingResponse(generate(), media_type="text/event-stream")


# EXPORT — summarise conversation and return as a downloadable .md file
class ExportRequest(BaseModel):
    messages: list[Message]
    model: str = settings.default_text_model


@app.post("/api/chat/export")
def chat_export(req: ExportRequest) -> FileResponse:
    """
    Ask the LLM to produce a structured summary of the conversation,
    then return it as a downloadable Markdown file.
    """
    if not req.messages:
        raise HTTPException(status_code=400, detail="Aucun message à exporter.")

    # Build a readable transcript for the LLM
    transcript_lines = []
    for m in req.messages:
        role_label = "Utilisateur" if m["role"] == "user" else "Assistant"
        content = m["content"] if isinstance(m["content"], str) else "[fichier joint]"
        transcript_lines.append(f"**{role_label}** : {content}")
    transcript = "\n\n".join(transcript_lines)

    summary_prompt = (
        "Voici une conversation entre un utilisateur et un assistant IA.\n\n"
        f"{transcript}\n\n"
        "Rédige un résumé structuré de cette conversation en Markdown, avec :\n"
        "- Un titre\n"
        "- Les points clés abordés\n"
        "- Les décisions ou conclusions importantes\n"
        "- Un résumé final en 2-3 phrases"
    )

    try:
        response = client.chat.completions.create(
            model=req.model,
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.3,
        )
        summary_md = response.choices[0].message.content
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Write to a temp file and serve it
    out_path = settings.tmp_dir / f"export_{uuid.uuid4()}.md"
    out_path.write_text(summary_md, encoding="utf-8")
    logger.info("Chat exported to %s", out_path)

    return FileResponse(
        str(out_path),
        media_type="text/markdown",
        filename="résumé_conversation.md",
    )

# FILE DOWNLOAD — serve generated files
@app.get("/api/files/{filename}")
def download_file(filename: str) -> FileResponse:
    """Serve a previously generated file for download."""
    # Security: only allow files from the generated_files directory
    safe_name = Path(filename).name
    file_path = settings.files_dir / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Fichier introuvable.")
    return FileResponse(str(file_path), filename=safe_name)


# MODERATION — standalone endpoint
@app.post("/api/moderate")
def moderate(req: ModerateRequest) -> dict:
    """Returns only flagged status — scores/categories stay server-side."""
    try:
        result = moderate_text(req.text)
        return {"flagged": result.flagged}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc