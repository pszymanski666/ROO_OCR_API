Okay, here is the English translation of the Product Requirements Document:

---

# **Product Requirements Document: PDF OCR API**

## Overview

**Problem:** Manual or semi-automatic text extraction from multi-page PDF documents is time-consuming, error-prone, and difficult to scale. There is a lack of a simple-to-use, automated API solution that can process large PDF files, report progress, and return accurate text.

**Solution:** A REST API built with Python (FastAPI) that accepts PDF files, utilizes the Tesseract OCR engine (with image enhancement via OpenCV) to extract text, processes tasks asynchronously (handling long operations and large files), provides a job progress tracking mechanism, implements result caching for identical files, and returns the processed text.

**Target Audience:** Application developers, document processing systems, companies needing to automate data extraction from PDFs.

**Value:** Automation and acceleration of the OCR process for PDF files, increased accuracy through image processing, capability for integration with other systems, scalability of task handling, resource savings through caching.

## Core Features

1.  **PDF File Upload**
    *   **What it does:** Allows users to securely upload PDF files for processing by the API.
    *   **Why it's important:** This is the primary entry point for the data to be processed.
    *   **How it works (general):** An API endpoint (e.g., `POST /upload`) accepts `multipart/form-data` requests containing a PDF file. The API validates the file type (`application/pdf`) and its size (within defined security limits). The file is temporarily saved in a dedicated directory.

2.  **Asynchronous OCR Processing**
    *   **What it does:** Performs the OCR process on the uploaded PDF file in the background, without blocking the user's response. This includes converting PDF pages to images, enhancing images using OpenCV, and extracting text using Tesseract.
    *   **Why it's important:** OCR processing, especially for large files, can be time-consuming. Asynchronicity allows the API to remain responsive and handle multiple tasks concurrently. Image enhancement increases OCR accuracy.
    *   **How it works (general):** After saving the file, the API creates a task in a task queue system (Celery) with a unique identifier. A Celery worker picks up the task, converts the PDF to images (`pdf2image`), processes each image using OpenCV (e.g., grayscale conversion, adaptive thresholding, noise removal, potential deskewing), and then passes the enhanced image to Tesseract (`pytesseract`) for text extraction (with default `--psm` and `--oem`). Results from all pages are aggregated.

3.  **Job Progress Tracking**
    *   **What it does:** Allows users to check the status and progress of their OCR processing job.
    *   **Why it's important:** For long-running tasks, it's crucial to inform the user about what's happening (e.g., "queued", "processing X%", "completed", "error").
    *   **How it works (general):** An API endpoint (e.g., `GET /status/{task_id}`) accepts the task identifier. The API queries the Celery backend (or a dedicated status repository, e.g., Redis) to get the current state (`status`) and percentage progress (`progress`), which is updated by the Celery worker as it processes subsequent pages.

4.  **Result Retrieval**
    *   **What it does:** Allows users to retrieve the extracted text after the job has successfully completed.
    *   **Why it's important:** This is the final product of the API's operation – delivering the processed text.
    *   **How it works (general):** An API endpoint (e.g., `GET /result/{task_id}`) accepts the task identifier. If the task status is "COMPLETED", the API retrieves the aggregated text from the Celery result backend (or a dedicated result storage location, e.g., Redis or a file) and returns it to the user (e.g., as `text/plain` or within JSON).

5.  **Result Caching**
    *   **What it does:** Stores the results of successfully processed files. If a user uploads a file identical to one already processed, the API returns the cached result immediately, without reprocessing.
    *   **Why it's important:** Saves computational resources and time, providing instantaneous responses for repetitive requests.
    *   **How it works (general):** During file upload (`POST /upload`), the API calculates a cryptographic hash (e.g., SHA-256) of the file's content. It checks a dedicated cache (e.g., Redis) to see if a result already exists for this hash. If yes, it immediately returns a status indicating completion and the task ID leading to the cached result. If not, it proceeds with normal processing, and upon successful completion, saves the result in the cache under the file hash key.

6.  **Temporary File Management**
    *   **What it does:** Manages the storage of uploaded PDF files and any intermediate image files in a dedicated temporary location. Automatically cleans up old files.
    *   **Why it's important:** Prevents the disk from filling up with files that are no longer needed.
    *   **How it works (general):** Uploaded PDF files are saved in a configured `temp_files/` directory. A separate, scheduled process (e.g., a Celery Beat task, cronjob, systemd timer) runs periodically (e.g., once a day) and deletes files from `temp_files/` that are older than a specified time (e.g., 24 hours), ensuring it doesn't delete files associated with active tasks.

7.  **API Security**
    *   **What it does:** Implements basic security measures to protect the API.
    *   **Why it's important:** Protects the API from abuse, DoS attacks, and ensures data integrity.
    *   **How it works (general):** Includes input validation (file type, maximum file size), rate limiting at the API or individual endpoint level, potentially basic authentication (e.g., API key in the header) if required.

8.  **Detailed Error Logging**
    *   **What it does:** Records information about application operation, warnings, and errors in a structured manner.
    *   **Why it's important:** Crucial for debugging, monitoring application health, and diagnosing problems (e.g., Tesseract errors, file issues, conversion errors).
    *   **How it works (general):** Utilization of Python's `logging` module. Configuration of different logging levels (DEBUG, INFO, WARNING, ERROR, CRITICAL). Logging key events (request received, task start/end, OCR errors, file errors) with context (e.g., task ID, filename). Saving full tracebacks for exceptions. Logs can be directed to the console, files (with rotation), or a centralized logging system.

## User Experience

*   **User Persona:** Developer (Backend/Fullstack) integrating the OCR API into their application or workflow. Needs a simple, reliable way to extract text from PDFs without managing OCR infrastructure.
*   **Key User Flows:**
    1.  **File Processing:**
        *   Developer sends a `POST /upload` request with a PDF file.
        *   API responds with `202 Accepted` and returns a `task_id`.
        *   Developer periodically sends `GET /status/{task_id}` requests.
        *   API responds with the status (e.g., `QUEUED`, `PROCESSING`) and progress (`progress`).
        *   When the status is `COMPLETED`, the developer sends a `GET /result/{task_id}` request.
        *   API returns the extracted text.
    2.  **Cached File Processing:**
        *   Developer sends a `POST /upload` request with a PDF file that has been processed before.
        *   API (after checking the cache) responds with `202 Accepted` (or immediately with a status indicating a cache hit, e.g., a special field in the response) and returns the `task_id` associated with the existing result.
        *   Developer sends `GET /status/{task_id}` (optional, they might try fetching the result directly).
        *   API responds `COMPLETED`, `progress: 100`.
        *   Developer sends `GET /result/{task_id}`.
        *   API returns the cached text.
    3.  **Error Handling:**
        *   Developer sends a `POST /upload` request.
        *   API returns a `task_id`.
        *   An error occurs during processing (e.g., corrupted PDF, Tesseract error).
        *   Developer sends `GET /status/{task_id}`.
        *   API responds with status `FAILED` and an optional error message (`message`).
        *   Attempting to retrieve the result (`GET /result/{task_id}`) will result in an error (e.g., 404 or 409 with failure information).
*   **UI/UX Considerations:** As this is an API, the main "interface" is its definition (e.g., OpenAPI/Swagger). It should be:
    *   **Clear and Consistent:** Endpoint names, parameters, response schemas must be intuitive.
    *   **Well-Documented:** Descriptions of endpoints, parameters, possible responses, and error codes.
    *   **Predictable:** Consistent response formats for success and errors.

## Technical Architecture

*   **System Components:**
    *   **Web Framework:** FastAPI (Python) - for handling HTTP requests, validation, routing.
    *   **Web Server:** Uvicorn (or another ASGI server) - for running the FastAPI application.
    *   **Task Queue:** Celery (Python) - for managing and executing long-running OCR tasks in the background.
    *   **Message Broker:** Redis (or RabbitMQ) - for communication between FastAPI and Celery workers.
    *   **Result Backend:** Redis (or another Celery-supported backend) - for storing the status, progress, and results of Celery tasks.
    *   **OCR Engine:** Tesseract OCR (installed on the system, with required language packs, e.g., `pol`, `eng`).
    *   **Tesseract Wrapper:** Pytesseract (Python) - for interacting with Tesseract from Python.
    *   **PDF Handling:** `pdf2image` (Python, requires Poppler) - for converting PDF pages to images.
    *   **Image Processing:** OpenCV (`opencv-python-headless`) - for preprocessing images before OCR.
    *   **Cache Store:** Redis - for storing file hashes and associated results.
    *   **Logging:** Python's standard `logging` module.
    *   **Scheduling (for cleanup):** Celery Beat (if using Celery for everything) or a system `cron`/`systemd timer`.
*   **Data Models (Pydantic Schemas):**
    *   `JobStatus`: (`task_id: str`, `status: str` [e.g., QUEUED, PROCESSING, COMPLETED, FAILED], `progress: int` [0-100], `message: Optional[str]`)
    *   `UploadResponse`: (`task_id: str`, `status: str` [e.g., QUEUED or CACHED], `message: Optional[str]`)
    *   `ErrorResponse`: (`detail: str`)
*   **APIs and Integrations:**
    *   `POST /upload`: Accepts `UploadFile`, returns `UploadResponse` (status 202).
    *   `GET /status/{task_id}`: Accepts `task_id`, returns `JobStatus` (status 200) or `ErrorResponse` (e.g., 404).
    *   `GET /result/{task_id}`: Accepts `task_id`, returns `text/plain` with the result (status 200) or `ErrorResponse` (e.g., 404, 409).
    *   **Internal:** FastAPI -> Redis (Broker), Celery Worker <-> Redis (Broker, Backend), Celery Worker -> Tesseract, Celery Worker -> OpenCV, Celery Worker -> pdf2image, FastAPI/Celery Worker -> Redis (Cache).
*   **Infrastructure Requirements:**
    *   Python runtime environment (e.g., 3.8+).
    *   Installed Tesseract OCR and required language packs.
    *   Installed Poppler (for `pdf2image`).
    *   Running Redis server (for broker, backend, and cache).
    *   Disk space for temporary PDF and image files (`temp_files/`).
    *   System to run the API server (Uvicorn) and Celery workers.
    *   System to run the cleanup task (if not Celery Beat).

## Development Roadmap (Scope-focused, No Timelines)

**Phase 1: Foundations and Core OCR (MVP)**

1.  **Environment Setup:** Project setup, virtual environment, installation of basic dependencies (FastAPI, Uvicorn, Celery, Redis client, Pytesseract, pdf2image, Pillow). System installation of Tesseract and Poppler.
2.  **Celery Configuration:** Basic Celery setup with Redis as broker and backend.
3.  **`/upload` Endpoint (Basic):** Accept PDF file, save temporarily (in `temp_files/`), validate file type, dispatch Celery task, return `task_id`.
4.  **Celery OCR Task (Basic):** Convert PDF to images, simple loop over images, execute OCR via `pytesseract` (default options), aggregate text, store result in Celery backend.
5.  **`/status` Endpoint (Basic):** Read task status from Celery backend (PENDING, STARTED, SUCCESS, FAILURE).
6.  **`/result` Endpoint (Basic):** Retrieve result from Celery backend for completed tasks.
7.  **Basic Error Handling:** Catch exceptions in the Celery task and mark the task as FAILED.

**Phase 2: Process Improvements and Reliability**

8.  **Progress Tracking:** Implement `progress` updates in the Celery task (after processing each page or every N pages) and return it in `/status`.
9.  **OpenCV Image Enhancement:** Add image processing step using OpenCV (e.g., grayscale, thresholding) in the Celery task before passing to Tesseract.
10. **Detailed Logging:** Implement `logging` configuration for FastAPI and Celery, add logs for key operations and errors.
11. **Temporary File Management & Cleanup:** Create a dedicated `temp_files/` directory. Implement a cleanup mechanism (e.g., Celery Beat task or system script) to remove old files.

**Phase 3: Optimization and Security**

12. **Result Caching:** Implement file hash calculation in `/upload`, check/save results in Redis (cache), modify `/upload` and `/result` logic to handle caching.
13. **Basic Security:** Add maximum file size validation. Implement simple rate limiting for endpoints.
14. **Unit and Integration Tests:** Create a test suite (`pytest`) for:
    *   API endpoint logic (using `TestClient`).
    *   Key helper functions.
    *   Celery task logic (mocking external dependencies like Tesseract/OpenCV or using small test files).
    *   Error handling and edge cases.

**Phase 4: Further Enhancements (Future Enhancements)**

15. **Advanced Security:** Implement authentication/authorization (e.g., API keys, OAuth2).
16. **Advanced Image Processing:** Add more advanced OpenCV techniques (e.g., deskewing, line removal, adaptive thresholding) and potentially make them configurable.
17. **External Configuration:** Move settings (paths, Redis URL, limits) to environment variables or configuration files.
18. **User Interface (Optional):** A simple web UI for uploading files and monitoring status.
19. **Worker Scalability:** Configure and test running multiple Celery workers.
20. **More Sophisticated Error Handling:** Different API error codes for specific issues, more detailed error messages.

## Logical Dependency Chain

1.  **Phase 1 (Foundations):** Start by setting up the FastAPI application skeleton, integrating with Celery/Redis, and creating the basic flow: upload -> simple OCR task -> get status/result. This creates a working, albeit minimal, product. *Goal: Achieve a working end-to-end flow as quickly as possible.*
2.  **Phase 2 (Process Improvements):** On the working foundation, add key enhancements: progress tracking (critical for UX), image enhancement (critical for result quality), logging (essential for maintenance), and file management (essential for stability). These features build upon the existing Celery task and API structure.
3.  **Phase 3 (Optimization and Security):** After stabilizing the core and adding key improvements, focus on optimization (caching) and security (rate limiting, validation). Testing is crucial at this stage to ensure that new features and refactoring do not break existing logic.
4.  **Phase 4 (Further Enhancements):** These features are extensions and can be added as needed, building on an already robust and tested application.

*Each phase (and even each point within a phase) is intended as a relatively atomic unit of work that can be developed and tested.*

## Risks and Mitigations

*   **Risk:** Low OCR accuracy for certain document types.
    *   **Mitigation:** Implement flexible OpenCV preprocessing, test with diverse documents, potentially allow configuration of Tesseract/OpenCV parameters in the future. Log OCR quality/confidence scores (if Tesseract provides them).
*   **Risk:** Performance issues with very large PDF files or high load.
    *   **Mitigation:** Asynchronous architecture with Celery, optimize PDF-to-image conversion and OpenCV processing, ability to scale Celery workers, implement rate limiting and file size limits, use an efficient broker/backend (Redis). Monitor server and worker resources.
*   **Risk:** File handling errors (corrupted files, read/write issues, cleaning up active files).
    *   **Mitigation:** Robust file validation upon upload, handle `IOError` exceptions, implement a mechanism to check task status before deleting a file via the cleanup process (e.g., by checking in Redis if the task associated with the file is still active).
*   **Risk:** External dependencies (Tesseract, Poppler) - installation issues, compatibility, errors.
    *   **Mitigation:** Use containerization (e.g., Docker) to manage the environment and system dependencies. Thorough testing on the target environment. Handle errors originating from these tools within the Python code.
*   **Risk:** Insufficient API security.
    *   **Mitigation:** Implement validation, rate limiting (Phase 3), consider authentication (Phase 4). Regular security reviews, update dependencies.
*   **Risk:** Cache becomes inconsistent or too large.
    *   **Mitigation:** Use a strong hashing algorithm (SHA-256 minimizes collisions). Implement a cache eviction policy (e.g., TTL or LRU in Redis) if necessary. Monitor cache size.
*   **Risk:** Testing complexity (asynchronicity, external dependencies).
    *   **Mitigation:** Use `pytest` with plugins (e.g., `pytest-asyncio`), mock dependencies for unit tests, create integration tests covering the FastAPI-Celery-Redis flow in a controlled environment.

## Appendix

*   **Tesseract Configuration:** Use default Page Segmentation Mode (`--psm 3`) and OCR Engine Mode (`--oem 3`, if using LSTM). Polish language (`-l pol`) as default, potentially configurable in the future (add English `-l eng` etc.).
*   **OpenCV Pre-processing Steps (Proposed):**
    1.  Convert to grayscale (`cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`).
    2.  Apply adaptive thresholding (e.g., `cv2.adaptiveThreshold`) or global thresholding (Otsu `cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)`), depending on the image type.
    3.  (Optional) Noise removal (e.g., `cv2.medianBlur`).
    4.  (Optional, more advanced) Skew detection and correction (deskewing).
*   **File Hashing:** Use SHA-256 to generate the content hash of the file for caching purposes.

---