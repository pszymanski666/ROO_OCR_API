import os
import uuid
import hashlib
import os
import uuid
import hashlib
import redis.asyncio as redis # Import redis.asyncio
import os
import uuid
import hashlib
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Depends, Header, Request
from fastapi.responses import PlainTextResponse
from celery.result import AsyncResult
from schemas import UploadResponse, JobStatus, PageResult, OcrResultResponse, OcrTasksResponse # Import new schemas and OcrTasksResponse
from tasks import perform_ocr_task
import json # Import json
from settings import settings # Import settings
from db import insert_ocr_task, get_all_ocr_tasks, insert_cache_hit # Import insert_ocr_task, get_all_ocr_tasks, and insert_cache_hit

import logging
from enum import Enum
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Use settings for configuration
TEMP_DIR = settings.TEMP_FILES_DIR
redis_client = redis.Redis.from_url(settings.CELERY_BROKER_URL) # Use redis.asyncio.Redis

class OcrLanguage(str, Enum):
    afr = "afrikaans"
    sqi = "albański"
    amh = "amharski"
    eng = "angielski"
    ara = "arabski"
    asm = "asamski"
    aze_cyrl = "azerski_cyrylica"
    aze = "azerski_łacinka"
    eus = "baskijski"
    ben = "bengalski"
    bel = "białoruski"
    mya = "birmański"
    bos = "bośniacki"
    bre = "bretoński"
    bul = "bułgarski"
    ceb = "cebuański"
    chi_tra = "chiński_tradycyjny"
    chi_tra_vert = "chiński_tradycyjny_pionowy"
    chi_sim = "chiński_uproszczony"
    chi_sim_vert = "chiński_uproszczony_pionowy"
    hrv = "chorwacki"
    ces = "czeski"
    chr = "czirokeski"
    dan = "duński"
    dzo = "dzongkha"
    epo = "esperanto"
    est = "estoński"
    fao = "farerski"
    fil = "filipiński"
    fin = "fiński"
    fra = "francuski"
    fry = "fryzyjski"
    glg = "galicyjski"
    ell = "grecki"
    kat = "gruziński"
    guj = "gudżarati"
    hat = "haitański"
    heb = "hebrajski"
    hin = "hindi"
    spa = "hiszpański"
    ind = "indonezyjski"
    iku = "inuktitut"
    gle = "irlandzki"
    isl = "islandzki"
    jpn = "japoński"
    jpn_vert = "japoński_pionowy"
    jav = "jawajski"
    yid = "jidysz"
    yor = "joruba"
    kan = "kannada"
    cat = "kataloński"
    kaz = "kazachski"
    que = "keczua"
    khm = "khmerski"
    kir = "kirgiski"
    kor = "koreański"
    kor_vert = "koreański_pionowy"
    cos = "korsykański"
    kmr = "kurdyjski"
    lao = "laotański"
    lit = "litewski"
    ltz = "luksemburski"
    mkd = "macedoński"
    mal = "malajalam"
    msa = "malajski"
    div = "malediwski"
    mlt = "maltański"
    mri = "maoryski"
    mar = "marathi"
    mon = "mongolski"
    nep = "nepalski"
    nld = "holenderski"
    deu = "niemiecki"
    nor = "norweski"
    oci = "oksytański"
    ori = "orija"
    hye = "ormiański"
    pus = "paszto"
    pan = "pendżabski"
    fas = "perski_farsi"
    pol = "polski"
    por = "portugalski"
    rus = "rosyjski"
    ron = "rumuński"
    san = "sanskryt"
    srp = "serbski_cyrylica"
    srp_latn = "serbski_łacina"
    snd = "sindhi"
    grc = "starogrecki"
    kat_old = "starogruziński"
    spa_old = "starohiszpański"
    ita_old = "starowłoski"
    swa = "suahili"
    sun = "sundajski"
    sin = "syngaleski"
    syr = "syriacki"
    gla = "szkocki_galicki"
    swe = "szwedzki"
    slk = "słowacki"
    slk_frak = "słowacki_frak"
    slv = "słoweński"
    tgk = "tadżycki"
    tgl = "tagalog"
    tha = "tajski"
    tam = "tamilski"
    tat = "tatarski"
    tel = "telugu"
    tir = "tigrinia"
    ton = "tongijski"
    tur = "turecki"
    bod = "tybetański"
    uig = "ujgurski"
    ukr = "ukraiński"
    urd = "urdu"
    uzb_cyrl = "uzbecki_cyrylica"
    uzb = "uzbecki_łacina"
    cym = "walijski"
    vie = "wietnamski"
    hun = "węgierski"
    ita = "włoski"
    lat = "łaciński"
    lav = "łotewski"
    enm = "średnioangielski"
    frm = "średniofrancuski"


@app.on_event("startup")
async def startup_event():
    await FastAPILimiter.init(redis_client)

@app.on_event("shutdown")
async def shutdown_event():
    await FastAPILimiter.close()

# Dependency for API Key Authentication
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key not in settings.ALLOWED_API_KEYS:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")
    return x_api_key

@app.post("/upload", response_model=UploadResponse, dependencies=[Depends(RateLimiter(times=settings.RATE_LIMIT_PER_MINUTE, seconds=60)), Depends(verify_api_key)]) # Use settings for rate limiting and add auth dependency
async def upload_pdf(request: Request, file: UploadFile = File(...), language: OcrLanguage="angielski"):
    """
    Upload a PDF file for OCR processing with caching, file size validation, and API key authentication.
    The 'language' parameter specifies the language(s) for OCR.
    """
    logger.info(f"Received upload request for file: {file.filename} with language(s): {language.name}")
    if language.name != "pol":
        selected_language = f"{language.name}+eng"
    else:
        selected_language = language.name
    # Read file content for hashing and size check
    file_content = await file.read()
    await file.seek(0) # Reset file pointer after reading

    # Implement maximum file size validation using settings
    MAX_FILE_SIZE_BYTES = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(file_content) > MAX_FILE_SIZE_BYTES:
        logger.warning(f"File size exceeds limit: {len(file_content)} bytes")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds the maximum limit of {settings.MAX_FILE_SIZE_MB}MB."
        )

    # Calculate file hash (SHA-256)
    file_hash = hashlib.sha256(file_content).hexdigest()
    logger.info(f"Calculated file hash: {file_hash}")

    # Check cache for existing result
    cached_result = await redis_client.get(f"ocr_result:{file_hash}:{selected_language}") # Use await
    if cached_result:
        # Find the task_id associated with this hash
        cached_task_id = await redis_client.get(f"hash_task:{file_hash}") # Use await
        if cached_task_id:
            task_id_str = cached_task_id.decode()
            logger.info(f"Cache hit for hash {file_hash}. Returning cached result for task ID: {task_id_str}")
            # Insert a record into the cache hits table
            client_ip = request.client.host if request.client else "unknown"
            insert_cache_hit(task_id_str, client_ip)
            logger.info(f"Logged cache hit for task {task_id_str} in ocr_cache_hits table.")
            return UploadResponse(
                task_id=task_id_str,
                status="SUCCESS", # Indicate success as the task is done
                message="Task already processed. Result available via /result/{task_id} endpoint."
            )
        else:
             # This case should ideally not happen if hash_task and ocr_result are set together
             logger.warning(f"Cache hit for hash {file_hash} but no associated task_id found.")


    if file.content_type != "application/pdf":
        logger.warning(f"Invalid file type uploaded: {file.content_type}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF files are allowed.")

    # Save the uploaded file temporarily
    file_id = str(uuid.uuid4())
    temp_file_path = os.path.join(TEMP_DIR, f"{file_id}.pdf")
    try:
        # Ensure the temp directory exists
        os.makedirs(TEMP_DIR, exist_ok=True)
        with open(temp_file_path, "wb") as f:
            f.write(file_content) # Use the already read content
        logger.info(f"Saved uploaded file to: {temp_file_path}")
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save uploaded file.")


    # Dispatch the OCR task to Celery
    task = perform_ocr_task.delay(temp_file_path, file_hash, selected_language) # Pass file_hash and language to task
    logger.info(f"Dispatched OCR task with ID: {task.id} for language(s): {selected_language}")

    # Insert task into database
    client_ip = request.client.host if request.client else "unknown"
    insert_ocr_task(task.id, client_ip, file.filename, language_code=selected_language)
    logger.info(f"Inserted task {task.id} into database with language {selected_language}.")

    # Store hash-task_id mapping in Redis
    await redis_client.set(f"hash_task:{file_hash}", task.id) # Use await
    # Set an expiration for the hash-task_id mapping, e.g., 24 hours
    await redis_client.expire(f"hash_task:{file_hash}", 60 * 60 * 24) # Use await
    logger.info(f"Stored hash-task_id mapping in Redis: {file_hash} -> {task.id}")

    # Also store task_id -> hash mapping for result retrieval
    await redis_client.set(f"task_hash:{task.id}", file_hash) # Use await
    # Store task_id -> language mapping in Redis
    await redis_client.set(f"task_language:{task.id}", selected_language) # Use await
    await redis_client.expire(f"task_language:{task.id}", 60 * 60 * 24) # Same expiration as task_hash, use await
    logger.info(f"Stored task_id-language mapping in Redis: {task.id} -> {selected_language}")
    await redis_client.expire(f"task_hash:{task.id}", 60 * 60 * 24) # Same expiration as hash_task, use await
    logger.info(f"Stored task_id-hash mapping in Redis: {task.id} -> {file_hash}")


    return UploadResponse(task_id=task.id, status=task.status, message="OCR task dispatched.")

@app.get("/status/{task_id}", response_model=JobStatus, dependencies=[Depends(verify_api_key)]) # Add auth dependency
async def get_job_status(task_id: str):
    """
    Get the status of an OCR job.
    """
    logger.info(f"Received status request for task ID: {task_id}")
    task_result = AsyncResult(task_id)
    status = task_result.status
    progress = 0
    message = None

    if status == "SUCCESS":
        progress = 100
        message = "OCR task completed successfully."
        logger.info(f"Task {task_id} status: SUCCESS")
    elif status == "FAILURE":
        progress = 100 # Task failed, progress is irrelevant, maybe keep at last reported or 0? Let's keep at 100 for simplicity in this phase.
        # Provide more detailed error message from task info
        error_info = task_result.info
        message = error_info.get('exc_message', 'OCR task failed.') if isinstance(error_info, dict) else 'OCR task failed.'
        logger.error(f"Task {task_id} status: FAILURE - {message}")
    elif status == "PROGRESS":
        progress = task_result.info.get('percent', 0) if isinstance(task_result.info, dict) else 0
        current = task_result.info.get('current', 0) if isinstance(task_result.info, dict) else 0
        total = task_result.info.get('total', 0) if isinstance(task_result.info, dict) else 0
        message = f"Processing page {current} of {total}"
        logger.info(f"Task {task_id} status: PROGRESS - {progress}% ({current}/{total})")
    else:
        message = "OCR task is pending or unknown status."
        logger.info(f"Task {task_id} status: {status}")


    return JobStatus(task_id=task_id, status=status, progress=progress, message=message)

@app.get("/result/{task_id}", response_model=OcrResultResponse, dependencies=[Depends(verify_api_key)]) # Add auth dependency and change response_model
async def get_job_result(task_id: str):
    logger.info(f"[{task_id}] Entering get_job_result endpoint.") # Add this log
    """
    Get the result of a completed OCR job, checking cache first.
    """
    logger.info(f"[{task_id}] Received result request for task ID: {task_id}")

    # Try to get the file hash from the task_id mapping in Redis
    file_hash = await redis_client.get(f"task_hash:{task_id}") # Use await

    if file_hash:
        file_hash = file_hash.decode()
        # Check cache for the result using the file hash
        # Retrieve the language associated with this task ID
        language = await redis_client.get(f"task_language:{task_id}") # Use await
        if not language:
             logger.warning(f"Task ID {task_id} has hash {file_hash} but no associated language found.")
             # If language is missing, we cannot check the language-specific cache.
             # Proceed to check task result directly.
             task_result = AsyncResult(task_id)
             if task_result.status == "SUCCESS":
                 logger.info(f"Task {task_id} result: SUCCESS (language missing for cache check)")
                 # Assuming task_result.get() returns a list of dictionaries here
                 result_data = task_result.get()
                 return OcrResultResponse(
                     task_id=task_id,
                     status="SUCCESS",
                     message="OCR task completed successfully (language missing for cache check).",
                     pages=result_data # Pass the list of dictionaries directly
                 )
             elif task_result.status == "FAILURE":
                 logger.error(f"Task {task_id} result: FAILURE (language missing for cache check)")
                 error_info = task_result.info
                 detail_message = error_info.get('exc_message', 'OCR task failed.') if isinstance(error_info, dict) else 'OCR task failed.'
                 raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail_message)
             else:
                 logger.warning(f"Task {task_id} result: Not completed yet (status: {task_result.status}, language missing for cache check)")
                 raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="OCR task not completed yet.")

        language = language.decode()
        logger.info(f"Retrieved language for task ID {task_id}: {language}")

        # Check cache for the result using the file hash and language
        cached_result = await redis_client.get(f"ocr_result:{file_hash}:{language}") # Use await
        if cached_result:
            logger.info(f"[{task_id}] Cache hit for task ID {task_id} (hash {file_hash}). Returning cached result.")
            # Add logging for debugging cache retrieval
            logger.info(f"[{task_id}] Raw cached result type: {type(cached_result)}, value[:100]: {cached_result[:100]}")
            logger.info(f"[{task_id}] Decoded cached result[:100]: {cached_result.decode()[:100]}")
            # Decode and parse the cached JSON result
            result_data = json.loads(cached_result.decode())
            logger.info(f"[{task_id}] Parsed result_data type: {type(result_data)}, len: {len(result_data) if isinstance(result_data, list) else 'N/A'}")
            return OcrResultResponse(
                task_id=task_id,
                status="SUCCESS",
                message="Result retrieved from cache.",
                pages=result_data # Pass the parsed list of dictionaries
            )
        else:
             logger.warning(f"Hash found for task ID {task_id} ({file_hash}) but no cached result found.")


    # If file_hash is not found in Redis, it means either the task never started,
    # or the task_hash mapping expired. In either case, we cannot retrieve the result
    # via the hash-based cache lookup. We should not attempt to get the result
    # directly from Celery as it would include the full OCR data in logs.
    # Instead, we check the task status and inform the user if it's not successful.
    task_result = AsyncResult(task_id)
    if task_result.status != "SUCCESS":
        logger.warning(f"Task {task_id} result: Not completed yet or failed (status: {task_result.status}).")
        # Provide a more informative message based on status
        if task_result.status == "FAILURE":
             error_info = task_result.info
             detail_message = error_info.get('exc_message', 'OCR task failed.') if isinstance(error_info, dict) else 'OCR task failed.'
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"OCR task failed: {detail_message}")
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"OCR task not completed yet. Current status: {task_result.status}")

    # If we reach here, it means file_hash was not found in Redis, but the task status is SUCCESS.
    # This is an unexpected state if the task_hash mapping should always exist for successful tasks.
    # Log a warning and raise an error, as we cannot retrieve the result without the hash.
    logger.error(f"[{task_id}] Task status is SUCCESS but file_hash not found in Redis. Cannot retrieve result.")
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error: Cannot retrieve result without file hash.")

@app.get("/tasks_history", response_model=OcrTasksResponse, dependencies=[Depends(verify_api_key)])
async def get_all_tasks_history():
   """
   Get a list of all OCR tasks from the database, including cache hit count.
   """
   logger.info("Received request for all OCR tasks history.")
   tasks_data = get_all_ocr_tasks()
   return OcrTasksResponse(tasks=tasks_data)

@app.get("/supported_languages", dependencies=[Depends(verify_api_key)])
async def get_supported_languages():
    """
    Get a list of all supported OCR languages.
    """
    logger.info("Received request for supported languages.")
    languages = [{"code": lang.name, "name": lang.value} for lang in OcrLanguage]
    return languages