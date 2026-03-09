from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str

    # Defaults — LLM
    default_text_model: str = "gpt-4o-mini"
    default_temperature: float = 0.7
    system_message: str = (
        "You are a helpful assistant that can answer queries based on your knowledge. "
        "If you don't know an answer, say 'I don't know'."
    )

    # Defaults — Image
    default_image_model: str = "gpt-image-1"
    default_image_size: str = "1024x1024"

    # Defaults — Audio
    default_stt_model: str = "gpt-4o-transcribe"
    default_tts_model: str = "tts-1"
    default_tts_voice: str = "alloy"
    default_tts_speed: float = 1.0

    # Weather
    weather_api_key: str = ""

    # Storage
    tmp_dir: Path = Path("temporary_files")
    files_dir: Path = Path("generated_files")

    class Config:
        env_file = "../.env"


settings = Settings()
settings.tmp_dir.mkdir(exist_ok=True)
settings.files_dir.mkdir(exist_ok=True)