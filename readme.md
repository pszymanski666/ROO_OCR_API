# Roo OCR API

Projekt API do przetwarzania OCR.

## Instalacja

1. Sklonuj repozytorium:
   ```bash
   git clone <URL_TWOJEGO_REPOZYTORIUM>
   cd roo_ocr_api
   ```
2. Zainstaluj zależności:
   ```bash
   pip install -r requirements.txt
   ```

## Użycie

1. Uruchom aplikację FastAPI:
   ```bash
   uvicorn main:app --reload
   ```
2. Uruchom Celery worker:
   ```bash
   celery -A tasks worker -l info
   ```
3. Uruchom Celery beat (scheduler):
   ```bash
   celery -A tasks beat -l info
   ```
4. API będzie dostępne pod adresem `http://127.0.0.1:8000`.

## Struktura projektu

- `main.py`: Główny plik aplikacji FastAPI.
- `db.py`: Konfiguracja bazy danych.
- `schemas.py`: Modele danych (Pydantic).
- `tasks.py`: Zadania Celery.
- `config.py`, `settings.py`: Pliki konfiguracyjne.
- `requirements.txt`: Lista zależności.