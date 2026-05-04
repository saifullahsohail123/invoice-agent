# schema.py

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from PIL.Image import Image as PILImage

# ─── Line item within an invoice ───────────────────────────────────────────
class LineItem(BaseModel):
    description: str
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    total: Optional[float] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

# ─── Full extracted invoice ─────────────────────────────────────────────────
class ExtractedInvoice(BaseModel):
    vendor_name: Optional[str] = None
    vendor_address: Optional[str] = None
    vendor_phone: Optional[str] = None
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None         # ISO format: YYYY-MM-DD
    due_date: Optional[str] = None
    buyer_name: Optional[str] = None
    buyer_address: Optional[str] = None
    buyer_phone: Optional[str] = None
    payment_details: Optional[str] = None
    line_items: List[LineItem] = Field(default_factory=list)
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    tax_rate: Optional[str] = None
    total: Optional[float] = None
    currency: str = "USD"
    payment_terms: Optional[str] = None
    notes: Optional[str] = None

    # Confidence scores per field (0.0 to 1.0)
    field_confidence: Dict[str, float] = Field(default_factory=dict)
    low_confidence_fields: List[str] = Field(default_factory=list)
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

# ─── LangGraph agent state ───────────────────────────────────────────────────
class InvoiceState(BaseModel):
    # Input
    source_path: str                               # Original file path
    source_type: Optional[Literal["pdf", "image"]] = None           # Detected type
    images: List[str] = Field(default_factory=list)  # Base64 encoded JPEG strings
    current_page: int = 0                          # For multi-page PDFs
    file_hash: Optional[str] = None                # SHA-256 for duplicate checks

    # Classification
    is_invoice: Optional[bool] = None
    classification_reason: Optional[str] = None

    # Extraction
    raw_llm_output: Optional[str] = None
    extracted: Optional[ExtractedInvoice] = None
    retry_count: int = 0
    max_retries: int = 2

    # Human in the loop
    awaiting_human: bool = False
    human_corrections: Dict[str, Any] = Field(default_factory=dict)
    human_approved: bool = False

    # Output
    status: Literal[
        "pending",
        "not_invoice",
        "extracting",
        "low_confidence",
        "awaiting_human",
        "complete",
        "failed"
    ] = "pending"
    error_message: Optional[str] = None
    db_row_id: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True  # Needed for PIL Image

# ─── Database row model ──────────────────────────────────────────────────────
class InvoiceDBRecord(BaseModel):
    id: Optional[int] = None
    source_path: str
    source_filename: str
    processed_at: datetime
    vendor_name: Optional[str]
    invoice_number: Optional[str]
    invoice_date: Optional[str]
    due_date: Optional[str]
    total: Optional[float]
    currency: str
    overall_confidence: float
    raw_json: str                 # Full extracted JSON stored as string
    status: str                   # "complete" | "human_approved" | "low_confidence"
