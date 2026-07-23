"""Request/response schemas for the RAG Document Portal API."""

from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    source_id: str
    filename: str
    num_pages: int
    num_chunks: int


class UploadResponse(BaseModel):
    source_id: str
    status: str  # "processing" (background job started) or "done" (dedup hit, no work needed)
    already_indexed: bool
    document: DocumentInfo | None = None  # populated when status == "done"


class UploadStatusResponse(BaseModel):
    source_id: str
    status: str  # "processing" | "done" | "error"
    document: DocumentInfo | None = None
    error: str | None = None


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question")
    document_ids: list[str] = Field(default_factory=list, description="Documents this session can query")
    session_id: str | None = Field(default=None, description="Omit on the first turn; reuse it for follow-ups")


class ChatResponse(BaseModel):
    session_id: str
    answer: str


class CompareRequest(BaseModel):
    document_ids: list[str] = Field(..., min_length=2, description="At least 2 documents to compare")
    topic: str = Field(..., min_length=1, description="The topic to compare across documents")


class CompareResponse(BaseModel):
    result: str


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int
    model_ready: bool
