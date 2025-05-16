from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # API Settings
    API_KEY: str = "supersecretapikey" # Default for now, should be changed in .env
    API_KEY_NAME: str = "X-API-Key"
    ALLOWED_API_KEYS: List[str] = ["supersecretapikey"] # Hardcoded list for now

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
