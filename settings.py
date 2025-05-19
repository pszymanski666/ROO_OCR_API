from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional
from celery import Celery
from datetime import timedelta

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # API Settings
    API_KEY: str = "supersecretapikey" # Default for now, should be changed in .env
    API_KEY_NAME: str = "X-API-Key"
    ALLOWED_API_KEYS: List[str] = ["123"] # Hardcoded list for now

    # Celery Settings
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    # File Settings
    TEMP_FILES_DIR: str = "temp_files"
    MAX_FILE_SIZE_MB: int = 100 # Maximum file size in MB

    # Image Processing Settings (basic example)
    # Example: Sigma for Gaussian blur in skimage (standard deviation)
    IMAGE_PROCESSING_BLUR_KERNEL_SIZE: int = 5
    # Domyślna rozdzielczość DPI do konwersji PDF->obraz
    PDF_CONVERSION_DPI: int = 300
    # Domyślne parametry progowania adaptacyjnego
    ADAPTIVE_THRESHOLD_BLOCK_SIZE: int = 11 # Musi być nieparzysty
    ADAPTIVE_THRESHOLD_C: int = 5 # Stała odejmowana od średniej
    # Domyślny czas wygaśnięcia cache Redis (w sekundach)
    REDIS_CACHE_TTL_OCR: int = 7 * 24 * 60 * 60 # 7 dni
    REDIS_CACHE_TTL_TASK_MAP: int = 1 * 24 * 60 * 60 # 1 dzień
    RATE_LIMIT_PER_MINUTE: int = 100
    
    # LLM Settings
    VLLM_API_URL: str = "http://127.17.0.1:8000/v1"
    VLLM_MODEL_NAME: str = "/model"
    VLLM_API_KEY: Optional[str] = None
    VLLM_MAX_TOKENS: int = 4096
    VLLM_TEMPERATURE: float = 0.7
    VLLM_REQUEST_TIMEOUT: int = 300  # Timeout dla żądań do API vLLM (w sekundach)

settings = Settings()

# Celery App Configuration
celery_app = Celery(
    "ocr_api", # Można to też wziąć z settings, jeśli jest taka potrzeba
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

celery_app.conf.update(
    task_ignore_result=False,
    task_track_started=True,
    beat_schedule={
        'cleanup-temp-files-daily': {
            'task': 'tasks.cleanup_temp_files', # Upewnij się, że ścieżka do zadania jest poprawna
            'schedule': timedelta(hours=24), # Run daily
            'args': (),
        },
    }
)

# Allowed OCR Languages
ALLOWED_OCR_LANGUAGES = [
    "afr", "amh", "ara", "asm", "aze", "aze_cyrl", "bel", "ben", "bod", "bos", "bre", "bul", "cat", "ceb", "ces",
    "chi_sim", "chi_sim_vert", "chi_tra", "chi_tra_vert", "chr", "cos", "cym", "dan", "dan_frak", "deu", "deu_frak",
    "deu_latf", "div", "dzo", "ell", "eng", "enm", "epo", "equ", "est", "eus", "fao", "fas", "fil", "fin", "fra",
    "frm", "fry", "gla", "gle", "glg", "grc", "guj", "hat", "heb", "hin", "hrv", "hun", "hye", "iku", "ind", "isl",
    "ita", "ita_old", "jav", "jpn", "jpn_vert", "kan", "kat", "kat_old", "kaz", "khm", "kir", "kmr", "kor",
    "kor_vert", "lao", "lat", "lav", "lit", "ltz", "mal", "mar", "mkd", "mlt", "mon", "mri", "msa", "mya", "nep",
    "nld", "nor", "oci", "ori", "osd", "pan", "pol", "por", "pus", "que", "ron", "rus", "san", "sin", "slk",
    "slk_frak", "slv", "snd", "spa", "spa_old", "sqi", "srp", "srp_latn", "sun", "swa", "swe", "syr", "tam", "tat",
    "tel", "tgk", "tgl", "tha", "tir", "ton", "tur", "uig", "ukr", "urd", "uzb", "uzb_cyrl", "vie", "yid", "yor"
]
