import os
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from settings import celery_app
import cv2
import numpy as np
import redis
import traceback
from datetime import datetime, timedelta
from celery.result import AsyncResult
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFSyntaxError, PDFPageCountError
import json # Dodano import json
from db import update_ocr_task # Import update_ocr_task

import logging

from settings import settings # Import settings


logger = logging.getLogger(__name__)

# Configure Redis for caching
redis_client = redis.Redis.from_url(settings.CELERY_BROKER_URL) # Use URL from settings

@celery_app.task(bind=True)
def perform_ocr_task(self,
                     pdf_path: str,
                     file_hash: str,
                     language: str = 'pol+eng',
                     dpi: int = settings.PDF_CONVERSION_DPI,
                     deskew: bool = False,
                     remove_lines: bool = False,
                     blur_kernel_size: int = settings.IMAGE_PROCESSING_BLUR_KERNEL_SIZE,
                     adaptive_thresh_block_size: int = settings.ADAPTIVE_THRESHOLD_BLOCK_SIZE,
                     adaptive_thresh_c: int = settings.ADAPTIVE_THRESHOLD_C):
    """
    Wykonuje OCR na pliku PDF z zaawansowanym przetwarzaniem obrazu, buforuje wynik i zwraca listę danych stron.

    Args:
        pdf_path (str): Ścieżka do pliku PDF.
        file_hash (str): Unikalny hash pliku (używany do klucza cache).
        language (str): Język(i) dla OCR (np. 'eng', 'pol', 'pol+eng').
        dpi (int): Rozdzielczość (dots per inch) używana podczas konwersji PDF na obrazy.
                   Wyższe wartości (np. 300) generalnie poprawiają jakość OCR.
        deskew (bool): Czy próbować automatycznie prostować przekrzywione obrazy stron.
        remove_lines (bool): Czy próbować usuwać linie poziome i pionowe.
        blur_kernel_size (int): Rozmiar jądra dla filtru Gaussian Blur (musi być nieparzysty i dodatni).
                                Używany do redukcji szumu przed progowaniem.
        adaptive_thresh_block_size (int): Rozmiar sąsiedztwa dla progowania adaptacyjnego (musi być nieparzysty i dodatni).
        adaptive_thresh_c (int): Stała odejmowana od średniej ważonej w progowaniu adaptacyjnym.

    Returns:
        list: Lista słowników, gdzie każdy słownik reprezentuje stronę z polami 'page_number', 'text', 'confidence'.

    Raises:
        FileNotFoundError: Jeśli plik PDF nie zostanie znaleziony.
        PDFInfoNotInstalledError: Jeśli brakuje zależności poppler.
        PDFPageCountError: Jeśli wystąpi błąd podczas liczenia stron PDF.
        PDFSyntaxError: Jeśli plik PDF jest uszkodzony lub ma nieprawidłową składnię.
        pytesseract.TesseractNotFoundError: Jeśli Tesseract nie jest zainstalowany lub nie ma go w PATH.
        Exception: Inne nieoczekiwane błędy podczas przetwarzania.
    """
    task_id = self.request.id
    logger.info(f"[{task_id}] Starting OCR task for PDF: {pdf_path}, Hash: {file_hash}, Lang: {language}")
    logger.info(f"[{task_id}] Options: DPI={dpi}, Deskew={deskew}, RemoveLines={remove_lines}, BlurKernel={blur_kernel_size}, ThreshBlock={adaptive_thresh_block_size}, ThreshC={adaptive_thresh_c}")

    # --- 1. Wstępne sprawdzenie i konfiguracja ---
    if not os.path.exists(pdf_path):
        logger.error(f"[{task_id}] File not found: {pdf_path}")
        self.update_state(
            state='FAILURE',
            meta={'exc_type': 'FileNotFoundError', 'exc_message': f"PDF file not found at path: {pdf_path}"}
        )
        # Nie ma sensu ponawiać, jeśli plik nie istnieje
        # raise Ignore() # Można użyć, jeśli nie chcemy, by Celery oznaczał to jako błąd retry
        raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")

    # Upewnij się, że parametry przetwarzania obrazu są poprawne
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1
        logger.warning(f"[{task_id}] Blur kernel size must be odd, adjusting to {blur_kernel_size}")
    if adaptive_thresh_block_size % 2 == 0:
        adaptive_thresh_block_size += 1
        logger.warning(f"[{task_id}] Adaptive threshold block size must be odd, adjusting to {adaptive_thresh_block_size}")

    images = []
    pages_data = [] # Zmieniono na listę słowników

    try:
        # --- 2. Konwersja PDF na obrazy ---
        try:
            logger.info(f"[{task_id}] Converting PDF to images with DPI={dpi}...")
            images = convert_from_path(pdf_path, dpi=dpi)
            total_pages = len(images)
            if total_pages == 0:
                 logger.warning(f"[{task_id}] PDF file seems empty or conversion yielded no images: {pdf_path}")
                 self.update_state(state='SUCCESS', meta={'result': 'PDF empty or no images converted', 'pages': 0})
                 return [] # Zwróć pustą listę
            logger.info(f"[{task_id}] Converted PDF to {total_pages} images.")
            self.update_state(state='STARTED', meta={'current': 0, 'total': total_pages, 'percent': 0})

        except (PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError) as e:
            logger.error(f"[{task_id}] Failed to convert PDF: {pdf_path} - {e}", exc_info=True)
            self.update_state(
                state='FAILURE',
                meta={'exc_type': type(e).__name__, 'exc_message': str(e), 'traceback': traceback.format_exc()}
            )
            raise # Przekaż błąd dalej do Celery

        # --- 3. Przetwarzanie każdej strony ---

        # Ensure output directory exists for processed images
        # output_dir = "processed_images"
        # if not os.path.exists(output_dir):
        #     try:
        #         os.makedirs(output_dir)
        #         logger.info(f"[{task_id}] Created output directory: {output_dir}")
        #     except OSError as e:
        #         logger.error(f"[{task_id}] Failed to create output directory {output_dir}: {e}", exc_info=True)
                # Decide if this is a critical error or just log and continue
                # For now, log and continue, saving might fail later

        for i, image in enumerate(images):
            page_num = i + 1
            logger.info(f"[{task_id}] Processing page {page_num}/{total_pages}")

            try:
                # Konwersja PIL Image do formatu OpenCV (NumPy array BGR)
                img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                # Konwersja do skali szarości
                gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

                # --- Zaawansowane przetwarzanie obrazu ---

                # 1. Deskewing (Opcjonalne)
                if deskew:
                    try:
                        logger.info(f"[{task_id}] Performing OSD (Orientation Detection) for page {page_num}...")
                        # Użyj obrazu w skali szarości dla OSD.
                        # Można rozważyć użycie oryginalnego obrazu (img_cv), jeśli OSD zawodzi na szarym.
                        # Podanie języka może pomóc OSD.
                        osd = pytesseract.image_to_osd(gray_img, lang=language, output_type=pytesseract.Output.DICT)

                        # 'rotate' to kąt (0, 90, 180, 270), o który należy obrócić obraz, aby był poprawnie zorientowany.
                        # 'orientation' to wykryta orientacja (0=góra, 1=prawo, 2=dół, 3=lewo)
                        rotation_angle = osd.get('rotate', 0)
                        orientation_conf = osd.get('orientation_conf', 0.0)
                        logger.info(f"[{task_id}] OSD result for page {page_num}: Detected Orientation={osd.get('orientation', 'N/A')}, Required Rotation={rotation_angle}, Confidence={orientation_conf:.2f}")

                        # Zastosuj rotację tylko jeśli jest niezerowa i pewność jest wystarczająco wysoka
                        # Możesz dostosować próg pewności (np. > 1.0)
                        min_osd_confidence = 1.0
                        if rotation_angle != 0 and orientation_conf >= min_osd_confidence:
                            rotated_img = None
                            if rotation_angle == 90:
                                # Obrót o 90 stopni przeciwnie do ruchu wskazówek zegara
                                rotated_img = cv2.rotate(gray_img, cv2.ROTATE_90_CLOCKWISE)
                                logger.info(f"[{task_id}] Rotating page {page_num} by 90 degrees counter-clockwise based on OSD.")
                            elif rotation_angle == 180:
                                # Obrót o 180 stopni
                                rotated_img = cv2.rotate(gray_img, cv2.ROTATE_180)
                                logger.info(f"[{task_id}] Rotating page {page_num} by 180 degrees based on OSD.")
                            elif rotation_angle == 270:
                                # Obrót o 270 stopni przeciwnie do ruchu wskazówek zegara = 90 zgodnie ze wskazówkami zegara
                                rotated_img = cv2.rotate(gray_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                logger.info(f"[{task_id}] Rotating page {page_num} by 90 degrees clockwise (OSD requested 270) based on OSD.")
                            else:
                                logger.warning(f"[{task_id}] OSD returned unexpected rotation angle {rotation_angle} for page {page_num}. Skipping rotation.")

                            if rotated_img is not None:
                                gray_img = rotated_img # Zaktualizuj obraz do dalszego przetwarzania
                        elif rotation_angle != 0 and orientation_conf < min_osd_confidence:
                            logger.warning(f"[{task_id}] OSD rotation angle is {rotation_angle} but confidence ({orientation_conf:.2f}) is below threshold ({min_osd_confidence}). Skipping rotation for page {page_num}.")
                        else:
                            logger.info(f"[{task_id}] Page {page_num} orientation seems correct according to OSD or confidence too low. No rotation applied.")

                    except pytesseract.TesseractNotFoundError:
                        logger.error(f"[{task_id}] Tesseract is not installed or not in PATH. Cannot perform OSD.", exc_info=False)
                        # Nie można kontynuować bez tesseracta, zgłoś błąd
                        self.update_state(state='FAILURE', meta={'exc_type': 'TesseractNotFoundError', 'exc_message': 'Tesseract not installed or not in PATH.'})
                        raise # Przerwij zadanie
                    except pytesseract.TesseractError as e:
                        # Czasem OSD może zawieść, ale OCR nadal może działać. Loguj jako ostrzeżenie.
                        logger.warning(f"[{task_id}] Tesseract OSD failed for page {page_num}: {e}. Skipping orientation correction.", exc_info=False)
                    except Exception as e:
                        # Inne błędy podczas OSD
                        logger.warning(f"[{task_id}] Error during OSD orientation correction for page {page_num}: {e}", exc_info=False)

                # 2. Redukcja szumu (Gaussian Blur)
                # Użyj skonfigurowanego rozmiaru jądra
                # if blur_kernel_size > 0:
                #     blur_img = cv2.GaussianBlur(gray_img, (blur_kernel_size, blur_kernel_size), 0)
                #     logger.info(f"[{task_id}] Applied Gaussian Blur with kernel size {blur_kernel_size} on page {page_num}.")
                # else:
                #     blur_img = gray_img # Pomiń rozmycie, jeśli kernel_size <= 0
                #     logger.info(f"[{task_id}] Skipping Gaussian Blur for page {page_num} (kernel size <= 0).")


                # 3. Progowanie adaptacyjne (Adaptive Thresholding)
                # Użyj skonfigurowanych parametrów
                # Uwaga: Dobór parametrów może wymagać eksperymentów dla różnych typów dokumentów.
                # Alternatywy: cv2.threshold z flagą cv2.THRESH_OTSU dla globalnego progowania Otsu.
                # thresh_img = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                #                                   cv2.THRESH_BINARY_INV, # Często _INV daje lepsze wyniki dla OCR (czarny tekst na białym tle)
                #                                   adaptive_thresh_block_size, adaptive_thresh_c)
                # logger.info(f"[{task_id}] Applied adaptive thresholding on page {page_num} (Block: {adaptive_thresh_block_size}, C: {adaptive_thresh_c}).")
                # --- NOWY KROK: Usuwanie szumu "sól" za pomocą operacji otwarcia ---
                # Zdefiniuj jądro. Rozmiar (np. 2x2 lub 3x3) zależy od wielkości kropek szumu.
                # Kształt może być prostokątny (MORPH_RECT) lub eliptyczny (MORPH_ELLIPSE). Elipsa często daje gładsze wyniki.
                # Zacznij od małego kernela, aby nie uszkodzić drobnych elementów tekstu (np. kropek nad i).
                # noise_kernel_size = (3, 3) # Eksperymentuj z (2, 2) lub (3, 3)
                # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, noise_kernel_size) # lub cv2.MORPH_ELLIPSE
                # # Zastosuj operację otwarcia. Jedna iteracja zwykle wystarcza.
                # thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=1)
                # logger.info(f"[{task_id}] Applied morphological opening to remove salt noise on page {page_num} with kernel {noise_kernel_size}.")
                # --- Koniec nowego kroku ---
                # 4. Usuwanie linii (Opcjonalne)
                # Uwaga: Może usuwać również fragmenty znaków (np. podkreślenia, l, i). Używać ostrożnie.
                if remove_lines:
                    try:
                        processed_lines = thresh_img.copy() # Pracuj na kopii
                        # Usuwanie linii poziomych
                        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1)) # Dobierz rozmiar kernela do grubości linii
                        detected_lines = cv2.morphologyEx(processed_lines, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
                        cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                        for c in cnts:
                            # Zamaluj wykryte linie na biało (lub czarno, jeśli nie użyto _INV w progowaniu)
                            cv2.drawContours(processed_lines, [c], -1, (0,0,0), 2) # Grubsze zamalowanie

                        # Usuwanie linii pionowych
                        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
                        detected_lines = cv2.morphologyEx(processed_lines, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
                        cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                        for c in cnts:
                            cv2.drawContours(processed_lines, [c], -1, (0,0,0), 2)

                        thresh_img = processed_lines # Zaktualizuj obraz po usunięciu linii
                        logger.info(f"[{task_id}] Attempted line removal on page {page_num}.")
                    except Exception as e:
                         logger.warning(f"[{task_id}] Could not perform line removal on page {page_num}: {e}", exc_info=False)

                # --- Koniec przetwarzania obrazu ---

                # Konwersja przetworzonego obrazu OpenCV z powrotem do PIL Image
                # Upewnij się, że format jest odpowiedni dla Pytesseract (zwykle skala szarości lub binarny)
                final_image_for_ocr = Image.fromarray(gray_img)

                # --- Save processed image for verification ---
                # try:
                #     output_filename = f"{task_id}_{file_hash}_page_{page_num}.png"
                #     output_path = os.path.join(output_dir, output_filename)
                #     final_image_for_ocr.save(output_path)
                #     logger.info(f"[{task_id}] Saved processed image for page {page_num} to {output_path}")
                # except Exception as save_err:
                #     logger.warning(f"[{task_id}] Failed to save processed image for page {page_num}: {save_err}", exc_info=False)
                    # Log the error but continue with OCR


                # --- 5. Wykonanie OCR ---
                try:
                    logger.info(f"[{task_id}] Performing Tesseract OCR on page {page_num} with lang='{language}'...")
                    # Dodaj konfigurację Tesseract, jeśli potrzebna, np. --psm 6 dla założenia jednolitego bloku tekstu
                    # custom_config = r'--oem 3 --psm 6' # Przykład
                    page_text = pytesseract.image_to_string(final_image_for_ocr, lang=language) #, config=custom_config)
                    logger.info(f"[{task_id}] OCR completed for page {page_num}.")

                    # Get detailed data for confidence calculation
                    page_data_dict = pytesseract.image_to_data(final_image_for_ocr, lang=language, output_type=pytesseract.Output.DICT)

                    # Calculate average word confidence
                    confidences = [int(conf) for conf in page_data_dict.get('conf', []) if int(conf) != -1]
                    average_confidence = sum(confidences) / len(confidences) if confidences else 0.0

                    # Create page data dictionary
                    page_entry = {
                        'page_number': page_num,
                        'text': page_text,
                        'confidence': round(average_confidence, 2) # Round for cleaner output
                    }
                    pages_data.append(page_entry) # Dodaj słownik strony do listy

                except pytesseract.TesseractNotFoundError:
                    logger.error(f"[{task_id}] Tesseract is not installed or not in PATH.", exc_info=True)
                    # Ten błąd jest krytyczny dla całego zadania
                    self.update_state(state='FAILURE', meta={'exc_type': 'TesseractNotFoundError', 'exc_message': 'Tesseract not installed or not in PATH.'})
                    raise # Przerwij zadanie
                except pytesseract.TesseractError as te:
                    logger.warning(f"[{task_id}] Tesseract error on page {page_num}: {te}", exc_info=False)
                    # Add an entry with error info if OCR fails for a page
                    page_entry = {
                        'page_number': page_num,
                        'text': f"[OCR Error: {te}]",
                        'confidence': 0.0 # Confidence is 0 if OCR failed
                    }
                    pages_data.append(page_entry)
                except Exception as ocr_err: # Inne potencjalne błędy z OCR
                    logger.warning(f"[{task_id}] Unexpected error during OCR on page {page_num}: {ocr_err}", exc_info=True)
                    # Add an entry with error info for unexpected errors
                    page_entry = {
                        'page_number': page_num,
                        'text': f"[OCR Error: Unexpected error occurred]",
                        'confidence': 0.0 # Confidence is 0 if OCR failed
                    }
                    pages_data.append(page_entry)


            except Exception as page_proc_err:
                # Błąd podczas przetwarzania pojedynczej strony - loguj i kontynuuj z następną
                logger.error(f"[{task_id}] Failed to process page {page_num}: {page_proc_err}", exc_info=True)
                # Add an entry with error info if page processing fails
                page_entry = {
                    'page_number': page_num,
                    'text': f"[Error processing page: {page_proc_err}]",
                    'confidence': 0.0 # Confidence is 0 if processing failed
                }
                pages_data.append(page_entry)
                # Można rozważyć aktualizację stanu z informacją o błędzie strony, ale nie przerywamy całego zadania

            # --- 6. Aktualizacja postępu ---
            percent_complete = int(((i + 1) / total_pages) * 100)
            self.update_state(state='PROGRESS', meta={'current': i + 1, 'total': total_pages, 'percent': percent_complete})
            logger.debug(f"[{task_id}] Progress update: {percent_complete}% ({i + 1}/{total_pages})")


        # --- 7. Cache'owanie wyniku w Redis ---
        cache_key = f"ocr_result:{file_hash}:{language}"
        try:
            # Serializuj listę słowników do JSON i zakoduj do bajtów przed zapisem w Redis
            redis_client.set(cache_key, json.dumps(pages_data).encode('utf-8'), ex=settings.REDIS_CACHE_TTL_OCR)
            logger.info(f"[{task_id}] Cached OCR result (pages_data) in Redis with key: {cache_key}")
        except Exception as redis_err:
            logger.warning(f"[{task_id}] Failed to cache OCR result (pages_data) in Redis: {redis_err}", exc_info=False)
            # Nie przerywaj zadania z powodu błędu cache

        # Mapowanie task_id -> hash (opcjonalne, ale przydatne do późniejszego odnajdywania wyników)
        task_map_key = f"task_hash:{task_id}"
        try:
            redis_client.set(task_map_key, file_hash, ex=settings.REDIS_CACHE_TTL_TASK_MAP)
            logger.info(f"[{task_id}] Stored task_id-hash mapping in Redis: {task_map_key} -> {file_hash}")
        except Exception as redis_err:
            logger.warning(f"[{task_id}] Failed to store task_id-hash mapping in Redis: {redis_err}", exc_info=False)

        # Oznacz jako sukces
        self.update_state(state='SUCCESS', meta={'result': 'OCR completed successfully', 'pages': total_pages})
        logger.info(f"[{task_id}] OCR task completed successfully for PDF: {pdf_path}")
        # Update task in database with success status and page count
        logger.info(f"[{task_id}] Attempting to update OCR task {task_id} in DB with status SUCCESS and pages {total_pages}.")
        update_ocr_task(task_id, 'SUCCESS', total_pages, language_code=language)
        logger.info(f"[{task_id}] Finished attempting to update OCR task {task_id} in DB.")

        # Zwróć listę pages_data
        # Return a simple success message instead of the full data to avoid logging large results
        return f"OCR task {task_id} completed successfully."

    except Exception as e:
        # Główny handler błędów - loguje, ustawia stan FAILURE i przekazuje wyjątek do Celery
        logger.error(f"[{task_id}] OCR task failed for PDF: {pdf_path} - {e}", exc_info=True)
        # Update task in database with failure status
        logger.error(f"[{task_id}] Attempting to update OCR task {task_id} in DB with status FAILURE.")
        update_ocr_task(task_id, 'FAILURE', language_code=language)
        logger.error(f"[{task_id}] Finished attempting to update OCR task {task_id} in DB.")
        self.update_state(
            state='FAILURE',
            meta={
                'exc_type': type(e).__name__,
                'exc_message': str(e),
                'traceback': traceback.format_exc(),
                'pdf_path': pdf_path,
                'file_hash': file_hash
            }
        )
        # Przekaż wyjątek, aby Celery oznaczyło zadanie jako nieudane
        # (i ewentualnie spróbowało ponownie, zgodnie z konfiguracją zadania)
        raise

    finally:
        # --- 8. Czyszczenie ---
        # Zawsze próbuj usunąć plik tymczasowy, jeśli istnieje
        # Sprawdź `processed_successfully` lub po prostu istnienie pliku
        if os.path.exists(pdf_path):
            try:
                logger.info(f"[{task_id}] Cleaning up temporary file: {pdf_path}")
                os.remove(pdf_path)
                logger.info(f"[{task_id}] Temporary file removed: {pdf_path}")
            except OSError as remove_err:
                logger.error(f"[{task_id}] Failed to remove temporary file {pdf_path}: {remove_err}", exc_info=True)
        else:
             # Jeśli plik został już usunięty lub wystąpił błąd przed jego utworzeniem/przetworzeniem
             logger.info(f"[{task_id}] Temporary file {pdf_path} not found for cleanup (already removed or error occurred earlier).")

# --- Helper function (optional, if needed elsewhere) ---
# Example: Function to convert kernel size to sigma (rough approximation)
# def kernel_to_sigma(ksize):
#    return 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
# Use settings for configuration
TEMP_DIR = settings.TEMP_FILES_DIR
CLEANUP_AGE_HOURS = 24 # This could also be moved to settings

# --- Helper function for LLM communication ---
import base64
import requests
import time

def send_images_to_llm(images, prompt=None):
    """
    Wysyła obrazy do modelu językowego poprzez API vLLM.

    Args:
        images (list): Lista obiektów PIL.Image.
        prompt (str, optional): Dodatkowy prompt dla modelu językowego.

    Returns:
        dict: Odpowiedź z modelu językowego.
    """
    logger.info(f"Sending {len(images)} images to vLLM API.")
    image_data = []
    for img in images:
        # Konwersja obrazu do formatu base64
        buffered = BytesIO()
        img.save(buffered, format="PNG") # Możesz wybrać inny format, np. JPEG
        img_str = base64.b64encode(buffered.getvalue()).decode()
        image_data.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}})

    # Przygotowanie treści żądania
    content = []
    if prompt:
        content.append({"type": "text", "text": prompt})
    content.extend(image_data)

    payload = {
        "model": settings.VLLM_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "max_tokens": settings.VLLM_MAX_TOKENS,
        "temperature": settings.VLLM_TEMPERATURE,
    }

    headers = {}
    if settings.VLLM_API_KEY:
        headers["Authorization"] = f"Bearer {settings.VLLM_API_KEY}"

    try:
        start_time = time.time()
        response = requests.post(f"{settings.VLLM_API_URL}/chat/completions", json=payload, headers=headers, timeout=settings.VLLM_REQUEST_TIMEOUT)
        response.raise_for_status() # Zgłoś wyjątek dla błędnych kodów statusu (4xx lub 5xx)
        end_time = time.time()
        processing_time = end_time - start_time

        response_data = response.json()
        logger.info(f"Received response from vLLM API. Processing time: {processing_time:.2f}s")

        # Przetworzenie odpowiedzi
        # Zakładamy, że odpowiedź jest w formacie zgodnym z OpenAI Chat Completions API
        llm_response_text = ""
        token_count = 0
        if response_data and response_data.get("choices"):
            choice = response_data["choices"][0]
            llm_response_text = choice.get("message", {}).get("content", "")
            token_count = response_data.get("usage", {}).get("total_tokens", 0)

        return {
            "model_name": settings.VLLM_MODEL_NAME,
            "response_text": llm_response_text,
            "processing_time": processing_time,
            "token_count": token_count
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"Error communicating with vLLM API: {e}", exc_info=True)
        raise # Przekaż błąd dalej
    except Exception as e:
        logger.error(f"Unexpected error processing vLLM response: {e}", exc_info=True)
        raise # Przekaż błąd dalej

# --- New Celery task for LLM processing ---
from io import BytesIO # Import BytesIO

@celery_app.task(bind=True)
def perform_llm_task(self,
                    pdf_path: str,
                    file_hash: str,
                    prompt: str = None,
                    dpi: int = settings.PDF_CONVERSION_DPI):
    """
    Przetwarza plik PDF, konwertuje go na obrazy stron i wysyła do modelu językowego.

    Args:
        pdf_path (str): Ścieżka do pliku PDF.
        file_hash (str): Unikalny hash pliku (używany do klucza cache).
        prompt (str, optional): Dodatkowy prompt dla modelu językowego.
        dpi (int): Rozdzielczość (dots per inch) używana podczas konwersji PDF na obrazy.

    Returns:
        dict: Słownik zawierający odpowiedź z modelu językowego i metadane.
    """
    task_id = self.request.id
    logger.info(f"[{task_id}] Starting LLM task for PDF: {pdf_path}, Hash: {file_hash}, Prompt: {prompt}")

    # --- 1. Wstępne sprawdzenie ---
    if not os.path.exists(pdf_path):
        logger.error(f"[{task_id}] File not found: {pdf_path}")
        self.update_state(
            state='FAILURE',
            meta={'exc_type': 'FileNotFoundError', 'exc_message': f"PDF file not found at path: {pdf_path}"}
        )
        raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")

    images = []
    llm_response_data = None

    try:
        # --- 2. Konwersja PDF na obrazy ---
        try:
            logger.info(f"[{task_id}] Converting PDF to images with DPI={dpi}...")
            images = convert_from_path(pdf_path, dpi=dpi)
            total_pages = len(images)
            if total_pages == 0:
                 logger.warning(f"[{task_id}] PDF file seems empty or conversion yielded no images: {pdf_path}")
                 self.update_state(state='SUCCESS', meta={'result': 'PDF empty or no images converted', 'pages': 0})
                 return {"message": "PDF empty or no images converted", "pages_processed": 0} # Zwróć pusty wynik
            logger.info(f"[{task_id}] Converted PDF to {total_pages} images.")
            self.update_state(state='STARTED', meta={'current': 0, 'total': total_pages, 'percent': 0})

        except (PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError) as e:
            logger.error(f"[{task_id}] Failed to convert PDF: {pdf_path} - {e}", exc_info=True)
            self.update_state(
                state='FAILURE',
                meta={'exc_type': type(e).__name__, 'exc_message': str(e), 'traceback': traceback.format_exc()}
            )
            raise # Przekaż błąd dalej do Celery

        # --- 3. Wysłanie obrazów do modelu językowego ---
        try:
            logger.info(f"[{task_id}] Sending images to LLM...")
            llm_response_data = send_images_to_llm(images, prompt)
            logger.info(f"[{task_id}] Received response from LLM.")
            self.update_state(state='PROGRESS', meta={'current': total_pages, 'total': total_pages, 'percent': 100, 'message': 'LLM processing complete'})

        except Exception as llm_err:
            logger.error(f"[{task_id}] Failed to send images to LLM or process response: {llm_err}", exc_info=True)
            self.update_state(
                state='FAILURE',
                meta={'exc_type': type(llm_err).__name__, 'exc_message': str(llm_err), 'traceback': traceback.format_exc()}
            )
            raise # Przekaż błąd dalej

        # --- 4. Cache'owanie wyniku w Redis ---
        cache_key = f"llm_result:{file_hash}:{prompt}" # Użyj hasha i promptu jako klucza cache
        try:
            # Serializuj wynik do JSON i zakoduj do bajtów przed zapisem w Redis
            redis_client.set(cache_key, json.dumps(llm_response_data).encode('utf-8'), ex=settings.REDIS_CACHE_TTL_OCR) # Możesz użyć innego TTL dla LLM
            logger.info(f"[{task_id}] Cached LLM result in Redis with key: {cache_key}")
        except Exception as redis_err:
            logger.warning(f"[{task_id}] Failed to cache LLM result in Redis: {redis_err}", exc_info=False)
            # Nie przerywaj zadania z powodu błędu cache

        # Mapowanie task_id -> hash (opcjonalne, ale przydatne do późniejszego odnajdywania wyników)
        task_map_key = f"llm_task_hash:{task_id}" # Użyj innego klucza dla zadań LLM
        try:
            redis_client.set(task_map_key, f"{file_hash}:{prompt or ''}", ex=settings.REDIS_CACHE_TTL_TASK_MAP) # Store hash and prompt
            logger.info(f"[{task_id}] Stored task_id-hash mapping in Redis: {task_map_key} -> {file_hash}")
        except Exception as redis_err:
            logger.warning(f"[{task_id}] Failed to store task_id-hash mapping in Redis: {redis_err}", exc_info=False)

        # Oznacz jako sukces
        self.update_state(state='SUCCESS', meta={'result': 'LLM processing completed successfully', 'pages_processed': total_pages, 'llm_response': llm_response_data})
        logger.info(f"[{task_id}] LLM task completed successfully for PDF: {pdf_path}")
        # Update task in database with success status and page count
        # logger.info(f"[{task_id}] Attempting to update LLM task {task_id} in DB with status SUCCESS and pages {total_pages}.")
        # update_llm_task(task_id, 'SUCCESS', total_pages, llm_response_data) # Potrzebna nowa funkcja update_llm_task
        # logger.info(f"[{task_id}] Finished attempting to update LLM task {task_id} in DB.")


        # Zwróć wynik
        return llm_response_data

    except Exception as e:
        # Główny handler błędów - loguje, ustawia stan FAILURE i przekazuje wyjątek do Celery
        logger.error(f"[{task_id}] LLM task failed for PDF: {pdf_path} - {e}", exc_info=True)
        # Update task in database with failure status
        # logger.error(f"[{task_id}] Attempting to update LLM task {task_id} in DB with status FAILURE.")
        # update_llm_task(task_id, 'FAILURE') # Potrzebna nowa funkcja update_llm_task
        # logger.error(f"[{task_id}] Finished attempting to update LLM task {task_id} in DB.")
        self.update_state(
            state='FAILURE',
            meta={
                'exc_type': type(e).__name__,
                'exc_message': str(e),
                'traceback': traceback.format_exc(),
                'pdf_path': pdf_path,
                'file_hash': file_hash
            }
        )
        # Przekaż wyjątek, aby Celery oznaczyło zadanie jako nieudane
        raise

    finally:
        # --- 5. Czyszczenie ---
        # Zawsze próbuj usunąć plik tymczasowy, jeśli istnieje
        if os.path.exists(pdf_path):
            try:
                logger.info(f"[{task_id}] Cleaning up temporary file: {pdf_path}")
                os.remove(pdf_path)
                logger.info(f"[{task_id}] Temporary file removed: {pdf_path}")
            except OSError as remove_err:
                logger.error(f"[{task_id}] Failed to remove temporary file {pdf_path}: {remove_err}", exc_info=True)
        else:
             logger.info(f"[{task_id}] Temporary file {pdf_path} not found for cleanup (already removed or error occurred earlier).")


@celery_app.task
def cleanup_temp_files():
    """
    Cleans up temporary PDF files older than CLEANUP_AGE_HOURS
    that are not associated with active or pending tasks.
    """
    logger.info("Starting temporary file cleanup task.")
    now = datetime.now()
    cleanup_threshold = now - timedelta(hours=CLEANUP_AGE_HOURS)

    if not os.path.exists(TEMP_DIR):
        logger.warning(f"Temporary directory not found: {TEMP_DIR}")
        return

    for filename in os.listdir(TEMP_DIR):
        if filename.endswith(".pdf"):
            file_path = os.path.join(TEMP_DIR, filename)
            try:
                # Extract task_id from filename (assuming filename is task_id.pdf)
                task_id = os.path.splitext(filename)[0]

                # Check task status
                task_result = AsyncResult(task_id)
                task_status = task_result.status

                if task_status in ["PENDING", "STARTED"]:
                    logger.info(f"Skipping cleanup for {filename}: Task {task_id} is {task_status}.")
                    continue

                # Check file modification time
                file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))

                if file_mtime < cleanup_threshold:
                    logger.info(f"Deleting old temporary file: {file_path} (Task status: {task_status})")
                    os.remove(file_path)
                else:
                    logger.info(f"Keeping recent temporary file: {file_path} (Task status: {task_status})")

            except FileNotFoundError:
                logger.warning(f"File not found during cleanup (already deleted?): {file_path}")
            except Exception as e:
                logger.error(f"Error during cleanup for file {file_path}: {e}", exc_info=True)

    logger.info("Temporary file cleanup task finished.")