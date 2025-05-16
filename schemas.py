from typing import Optional, Any
from pydantic import BaseModel
from typing import List
class UploadResponse(BaseModel):
    task_id: str
    status: str
    message: Optional[str] = None
    cached_result: Optional[str] = None

class JobStatus(BaseModel):
    task_id: str
    status: str
    progress: int
    message: Optional[str] = None

class ErrorResponse(BaseModel):
    detail: str
    # Optional field to include more specific error information if needed
    # For example, validation errors could include a list of issues
    errors: Optional[Any] = None

class PageResult(BaseModel):
   page_number: int
   text: str
   confidence: float

class OcrResultResponse(BaseModel):
   task_id: str
   status: str
   message: str
   pages: List[PageResult]

class OcrTask(BaseModel):
    id: str
    start_time: float
    end_time: Optional[float] = None
    client_ip: str
    document_name: str
    status: str
    page_count: Optional[int] = None
    language_code: Optional[str] = None
    cache_hits: int = 0 # Dodaj nowe pole dla liczby trafień w pamięci podręcznej

class OcrTasksResponse(BaseModel):
    tasks: List[OcrTask]