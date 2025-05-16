import sqlite3
import os
from datetime import datetime
import logging

DATABASE_PATH = 'ocr_tasks.db'

logger = logging.getLogger(__name__)

def create_connection():
    """Tworzy połączenie z bazą danych SQLite."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        return conn
    except sqlite3.Error as e:
        logger.error(f"Błąd połączenia z bazą danych: {e}")
        return None

def insert_ocr_task(task_id: str, client_ip: str, document_name: str, language_code: str = None):
    """Wstawia nowy rekord zadania OCR do bazy danych."""
    logger.info(f"[{task_id}] Attempting to insert OCR task into DB.")
    conn = create_connection()
    if conn:
        logger.info(f"[{task_id}] DB connection established for insert.")
        try:
            cursor = conn.cursor()
            start_time = datetime.now().timestamp()
            logger.info(f"[{task_id}] Executing INSERT for task {task_id}.")
            cursor.execute('''
                INSERT INTO ocr_tasks (id, start_time, client_ip, document_name, status, language_code)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (task_id, start_time, client_ip, document_name, 'STARTED', language_code))
            conn.commit()
            logger.info(f"[{task_id}] DB commit successful for insert task {task_id}.")
            print(f"Wstawiono zadanie OCR do bazy danych: {task_id}")
        except sqlite3.Error as e:
            logger.error(f"[{task_id}] Błąd podczas wstawiania zadania OCR {task_id}: {e}")
        finally:
            conn.close()
            logger.info(f"[{task_id}] DB connection closed after insert.")
    else:
        logger.error(f"[{task_id}] Failed to establish DB connection for insert task {task_id}.")


def update_ocr_task(task_id: str, status: str, page_count: int = None, language_code: str = None):
    """Aktualizuje rekord zadania OCR w bazie danych."""
    logger.info(f"[{task_id}] Attempting to update OCR task in DB: status={status}, page_count={page_count}, language_code={language_code}")
    conn = create_connection()
    if conn:
        logger.info(f"[{task_id}] DB connection established for update.")
        try:
            cursor = conn.cursor()
            end_time = datetime.now().timestamp()

            update_fields = {
                'end_time': end_time,
                'status': status
            }
            if page_count is not None:
                update_fields['page_count'] = page_count
            if language_code is not None:
                update_fields['language_code'] = language_code

            set_clause = ', '.join([f"{field} = ?" for field in update_fields.keys()])
            values = list(update_fields.values())
            values.append(task_id)

            logger.info(f"[{task_id}] Executing UPDATE for task {task_id} with fields: {list(update_fields.keys())}.")
            cursor.execute(f'''
                UPDATE ocr_tasks
                SET {set_clause}
                WHERE id = ?
            ''', values)

            conn.commit()
            logger.info(f"[{task_id}] DB commit successful for update task {task_id}.")
            print(f"Zaktualizowano zadanie OCR w bazie danych: {task_id} ze statusem {status}")
        except sqlite3.Error as e:
            logger.error(f"[{task_id}] Błąd podczas aktualizacji zadania OCR {task_id}: {e}")
        finally:
            conn.close()
            logger.info(f"[{task_id}] DB connection closed after update.")
    else:
        logger.error(f"[{task_id}] Failed to establish DB connection for update task {task_id}.")

def insert_cache_hit(task_id: str, client_ip: str):
    """Wstawia nowy rekord cache hit do bazy danych."""
    logger.info(f"[{task_id}] Attempting to insert cache hit into DB.")
    conn = create_connection()
    if conn:
        logger.info(f"[{task_id}] DB connection established for cache hit insert.")
        try:
            cursor = conn.cursor()
            hit_time = datetime.now().timestamp()
            logger.info(f"[{task_id}] Executing INSERT for cache hit for task {task_id}.")
            cursor.execute('''
                INSERT INTO ocr_cache_hits (task_id, hit_time, client_ip)
                VALUES (?, ?, ?)
            ''', (task_id, hit_time, client_ip))
            conn.commit()
            logger.info(f"[{task_id}] DB commit successful for cache hit for task {task_id}.")
            print(f"Zarejestrowano cache hit dla zadania: {task_id}")
        except sqlite3.Error as e:
            logger.error(f"[{task_id}] Błąd podczas wstawiania cache hit dla zadania {task_id}: {e}")
        finally:
            conn.close()
            logger.info(f"[{task_id}] DB connection closed after cache hit insert.")
    else:
        logger.error(f"[{task_id}] Failed to establish DB connection for cache hit for task {task_id}.")

def get_all_ocr_tasks():
    """Pobiera wszystkie rekordy zadań OCR z bazy danych."""
    logger.info("Attempting to get all OCR tasks from DB.")
    conn = create_connection()
    tasks = []
    if conn:
        logger.info("DB connection established for get_all_ocr_tasks.")
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            logger.info("Executing SELECT * FROM ocr_tasks.")
            cursor.execute('''
                SELECT
                    ot.*,
                    COUNT(och.task_id) AS cache_hits
                FROM
                    ocr_tasks ot
                LEFT JOIN
                    ocr_cache_hits och ON ot.id = och.task_id
                GROUP BY
                    ot.id
            ''')
            rows = cursor.fetchall()
            logger.info(f"Fetched {len(rows)} rows from ocr_tasks with cache hits.")
            for row in rows:
                task_data = dict(row)
                tasks.append(task_data)
            print(f"Pobrano {len(tasks)} zadań OCR z bazy danych z liczbą trafień w pamięci podręcznej.")
        except sqlite3.Error as e:
            logger.error(f"Błąd podczas pobierania zadań OCR: {e}")
        finally:
            conn.close()
            logger.info("DB connection closed after get_all_ocr_tasks.")
    else:
        logger.error("Failed to establish DB connection for get_all_ocr_tasks.")
    return tasks