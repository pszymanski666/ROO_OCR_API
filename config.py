from celery import Celery
from datetime import timedelta

CELERY_BROKER_URL = "redis://localhost:6379/0"
CELERY_RESULT_BACKEND = "redis://localhost:6379/0"

celery_app = Celery(
    "ocr_api",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND
)

celery_app.conf.update(
    task_ignore_result=False,
    task_track_started=True,
    beat_schedule={
        'cleanup-temp-files-daily': {
            'task': 'tasks.cleanup_temp_files',
            'schedule': timedelta(hours=24), # Run daily
            'args': (),
        },
    }
)

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

# Parametry dla modelu językowego
VLLM_API_URL = "http://localhost:8000/v1"  # Domyślny adres API vLLM
VLLM_MODEL_NAME = "google/gemma-3-27b"     # Nazwa modelu Gemma 3 27b
VLLM_API_KEY = "your-api-key"              # Klucz API (jeśli wymagany)
VLLM_MAX_TOKENS = 4096                     # Maksymalna liczba tokenów w odpowiedzi
VLLM_TEMPERATURE = 0.7                     # Temperatura dla generowania odpowiedzi