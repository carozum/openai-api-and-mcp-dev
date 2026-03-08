import os
import base64
import tempfile
from colorama import Fore
from pathlib import Path

import openai
from dotenv import load_dotenv
from pydub import AudioSegment

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()

SYSTEM_MESSAGE = "You are a helpful assistant that can answer queries based on your knowledge. If you don't know an answer, say 'I don't know'"


# =================================================
# SAVE FILE
# =================================================

def save_file(text, file_name):
    """Saves content to a file"""
    try:
        if not os.path.exists("temporary_files"):
            os.makedirs("temporary_files")
        if text is None:
            text = ""
        else:
            with open(file_name, "w", encoding="utf-8") as file:
                file.write(text)
                print(f"Content saved to {file_name}")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")


# =================================================
# AUDIO CONVERSION HELPER
# =================================================

def convert_to_wav(audio_path: str) -> str:
    """
    Convert any audio file to a clean 16kHz mono WAV.
    Returns the path to the converted file (in a temp dir).
    Handles WebM/OGG returned by st.audio_input in browsers.
    """
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio.export(tmp.name, format="wav")
    return tmp.name


# =================================================
# CHAT COMPLETIONS
# =================================================

def generate_chat_completion(
    prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
) -> str:
    """Generate a chat completion from a text prompt."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content


# =================================================
# CHAT COMPLETIONS WITH HISTORY (multi-turn)
# =================================================

def generate_chat_completion_with_history(
    messages: list,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    system_message: str = SYSTEM_MESSAGE,
) -> str:
    """
    Generate a chat completion from a full conversation history.
    `messages` is a list of {"role": "user"|"assistant", "content": str}.
    A system message is automatically prepended.
    """
    full_messages = [{"role": "system", "content": system_message}] + messages

    response = client.chat.completions.create(
        model=model,
        messages=full_messages,
        temperature=temperature,
    )
    return response.choices[0].message.content


# =================================================
# IMAGE GENERATION
# =================================================

def generate_image(
    prompt: str,
    model: str = "gpt-image-1",
    size: str = "1024x1024",
):
    """Generate an image from a prompt. Returns raw image bytes."""
    result = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
    )
    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)
    return image_bytes


# =================================================
# SPEECH TO TEXT
# =================================================

def speech_to_text(audio_path: str, model: str = "gpt-4o-transcribe") -> str:
    """
    Convert speech to text using OpenAI's Audio API.
    Converts audio to WAV first to ensure compatibility.
    """
    try:
        p = Path(audio_path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {audio_path}")

        # Convert to clean WAV before sending
        converted_path = convert_to_wav(audio_path)

        with open(converted_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
            )

        print(Fore.GREEN + f"Transcription: {transcript.text}" + Fore.RESET)
        return transcript.text

    except Exception as e:
        print(f"Error during transcription: {e}")
        raise e  # re-raise so Streamlit can display it


# =================================================
# SPEECH TO TRANSLATION
# =================================================

def speech_to_translation(audio_path: str) -> str:
    """
    Translate speech audio into English text.
    Always uses whisper-1 (only model supporting translations endpoint).
    """
    try:
        p = Path(audio_path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {audio_path}")

        # Convert to clean WAV before sending
        converted_path = convert_to_wav(audio_path)

        with open(converted_path, "rb") as audio_file:
            translation = client.audio.translations.create(
                model="whisper-1",  # only model available for translations
                file=audio_file,
            )

        print(Fore.GREEN + f"Translation: {translation.text}" + Fore.RESET)
        return translation.text

    except Exception as e:
        print(f"Error during translation: {e}")
        raise e


# =================================================
# TEXT TO SPEECH
# =================================================

def text_to_speech(
    text: str,
    output_path: str = "temporary_files/audio.wav",
    model: str = "tts-1",
    voice: str = "alloy",
    instructions: str = None,
    response_format: str = "wav",
    speed: float = 1.0,
) -> str:
    """Convert text to speech and save to output_path."""
    os.makedirs("temporary_files", exist_ok=True)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=text[:4096],
        instructions=instructions,
        response_format=response_format,
        speed=speed,
    ) as response:
        response.stream_to_file(out)

    return str(out)