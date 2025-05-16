import sqlite3
import os

db_path = 'ocr_tasks.db'

def create_database():
    """Tworzy plik bazy danych i tabelę ocr_tasks."""
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Utworzenie tabeli
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ocr_tasks (
                id TEXT PRIMARY KEY, -- Zmieniono na TEXT PRIMARY KEY do przechowywania task_id z Celery
                start_time REAL,
                client_ip TEXT,
                end_time REAL,
                status TEXT,
                page_count INTEGER,
                document_name TEXT,
                service_version TEXT,
                language_code TEXT
            )
        ''')

        # Utworzenie tabeli dla cache hitów
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ocr_cache_hits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT,
                hit_time REAL,
                client_ip TEXT,
                FOREIGN KEY (task_id) REFERENCES ocr_tasks (id)
            )
        ''')

        conn.commit()
        print(f"Baza danych '{db_path}' i tabele 'ocr_tasks', 'ocr_cache_hits' zostały utworzone pomyślnie.")

    except sqlite3.Error as e:
        print(f"Wystąpił błąd SQLite: {e}")
    except Exception as e:
        print(f"Wystąpił nieoczekiwany błąd: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    create_database()