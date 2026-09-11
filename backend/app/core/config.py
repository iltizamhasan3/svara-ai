from dataclasses import dataclass
import os


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


@dataclass(frozen=True, slots=True)
class Settings:
    environment: str = "development"
    max_dataset_rows: int = 10_000
    max_upload_bytes: int = 10 * 1024 * 1024
    max_csv_field_length: int = 100_000
    api_version: str = "0.1.0"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            environment=os.getenv("APP_ENV", "development"),
            max_dataset_rows=_int_env("MAX_DATASET_ROWS", 10_000),
            max_upload_bytes=_int_env("MAX_UPLOAD_BYTES", 10 * 1024 * 1024),
            max_csv_field_length=_int_env("MAX_CSV_FIELD_LENGTH", 100_000),
        )


settings = Settings.from_env()
