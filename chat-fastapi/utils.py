import base64
import logging
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import openai
from pydub import AudioSegment

from config import settings

# ── Logging ──────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── OpenAI client (singleton) ─────────────────────────────────────────────────
client = openai.OpenAI(api_key=settings.openai_api_key)


# ── Types ─────────────────────────────────────────────────────────────────────
Message = dict  # {"role": "user"|"assistant"|"system", "content": str}


# ── Helpers ───────────────────────────────────────────────────────────────────

@contextmanager
def tmp_audio_file(suffix: str = ".webm") -> Generator[Path, None, None]:
    """Context manager that yields a unique temp path and cleans it up on exit."""
    path = settings.tmp_dir / f"{uuid.uuid4()}{suffix}"
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def convert_to_wav(audio_path: Path) -> Path:
    """
    Convert any audio file to a clean 16 kHz mono WAV.
    Handles WebM/OGG returned by the browser MediaRecorder API.
    Returns a new temp file path (caller is responsible for cleanup).
    """
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(16_000).set_channels(1)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio.export(tmp.name, format="wav")
    logger.debug("Converted %s → %s", audio_path, tmp.name)
    return Path(tmp.name)


# ── LLM ──────────────────────────────────────────────────────────────────────

def generate_chat_completion(
    prompt: str,
    model: str = settings.default_text_model,
    temperature: float = settings.default_temperature,
) -> str:
    """Single-turn chat completion."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content


def generate_chat_completion_with_history(
    messages: list[Message],
    model: str = settings.default_text_model,
    temperature: float = settings.default_temperature,
    system_message: str = settings.system_message,
) -> str:
    """
    Multi-turn chat completion.
    `messages` is the full conversation history (user + assistant turns).
    A system message is automatically prepended.
    """
    full_messages = [{"role": "system", "content": system_message}] + messages
    response = client.chat.completions.create(
        model=model,
        messages=full_messages,
        temperature=temperature,
    )
    return response.choices[0].message.content


# ── Image ─────────────────────────────────────────────────────────────────────

def generate_image(
    prompt: str,
    model: str = settings.default_image_model,
    size: str = settings.default_image_size,
) -> bytes:
    """Generate an image from a prompt. Returns raw PNG bytes."""
    result = client.images.generate(model=model, prompt=prompt, size=size)
    image_bytes = base64.b64decode(result.data[0].b64_json)
    logger.debug("Image generated (%d bytes)", len(image_bytes))
    return image_bytes


# ── Speech to Text ────────────────────────────────────────────────────────────

def speech_to_text(
    audio_path: Path,
    model: str = settings.default_stt_model,
) -> str:
    """
    Transcribe audio to text.
    Converts to WAV first to ensure compatibility with the OpenAI API.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    wav_path = convert_to_wav(audio_path)
    try:
        with open(wav_path, "rb") as f:
            transcript = client.audio.transcriptions.create(model=model, file=f)
        logger.info("Transcription: %s", transcript.text)
        return transcript.text
    finally:
        wav_path.unlink(missing_ok=True)


# ── Speech to Translation ─────────────────────────────────────────────────────

def speech_to_translation(audio_path: Path) -> str:
    """
    Transcribe and translate audio to English.
    Only whisper-1 supports the translations endpoint.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    wav_path = convert_to_wav(audio_path)
    try:
        with open(wav_path, "rb") as f:
            translation = client.audio.translations.create(model="whisper-1", file=f)
        logger.info("Translation: %s", translation.text)
        return translation.text
    finally:
        wav_path.unlink(missing_ok=True)


# ── Text to Speech ────────────────────────────────────────────────────────────

def text_to_speech(
    text: str,
    output_path: Path,
    model: str = settings.default_tts_model,
    voice: str = settings.default_tts_voice,
    speed: float = settings.default_tts_speed,
    response_format: str = "wav",
) -> Path:
    """
    Synthesise text to speech and save to output_path.
    Returns the output path for chaining.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=text[:4096],
        response_format=response_format,
        speed=speed,
    ) as response:
        response.stream_to_file(output_path)
    logger.debug("TTS saved to %s", output_path)
    return output_path


# ── Moderation ────────────────────────────────────────────────────────────────

class ModerationResult:
    """Wraps an OpenAI moderation result with convenience helpers."""

    def __init__(self, flagged: bool, categories: dict[str, bool], scores: dict[str, float]):
        self.flagged = flagged
        self.categories = categories
        self.scores = scores

    @property
    def flagged_categories(self) -> list[str]:
        """Return names of categories that were flagged."""
        return [k for k, v in self.categories.items() if v]

    def __repr__(self) -> str:
        if not self.flagged:
            return "ModerationResult(ok)"
        return f"ModerationResult(flagged={self.flagged_categories})"


def moderate_text(text: str) -> ModerationResult:
    """
    Run OpenAI moderation on a text string.
    Returns a ModerationResult; does NOT raise — let the caller decide.
    """
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text,
    )
    r = result.results[0]
    mod = ModerationResult(
        flagged=bool(r.flagged),
        categories=dict(r.categories),
        scores=dict(r.category_scores),
    )
    if mod.flagged:
        logger.warning("Content flagged by moderation: %s", mod.flagged_categories)
    return mod