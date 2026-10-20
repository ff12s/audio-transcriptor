from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MEETING_NOTES_")

    model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str = "ru"
    # Кэш Hugging Face по умолчанию лежит на системном диске, поэтому модели хранятся здесь.
    model_dir: Path = Path("D:/models/whisper")
    # parents[2] — корень репозитория при editable-установке (uv sync).
    glossary_path: Path = Path(__file__).resolve().parents[2] / "glossary.txt"
