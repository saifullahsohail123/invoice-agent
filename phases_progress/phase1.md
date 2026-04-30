# Phase 1: Foundation & Environment (Completed)

This phase establishes the bedrock of the invoice-agent project. The files created in this phase define shared constants, data models (using Pydantic), the LLM model loading infrastructure, image preprocessing utilities, and the initial LangGraph input handler.

## Implementation Pipeline

### 1. `config.py`
**Purpose:** Stores configuration constants, prompt directory paths, thresholds, and VLM settings.
**Key Code Snippet:**
```python
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_DTYPE = "bfloat16"
MAX_IMAGE_LONG_SIDE = 1280   # pixels
CONFIDENCE_THRESHOLD = 0.80  # Fields below this trigger human-in-the-loop
```

### 2. `schema.py`
**Purpose:** Defines the data models. Uses Pydantic to ensure the input/output types of the AI match our rigorous demands.
**Key Code Snippet:**
```python
class ExtractedInvoice(BaseModel):
    vendor_name: Optional[str] = None
    invoice_number: Optional[str] = None
    total: Optional[float] = None
    currency: str = "USD"
    line_items: List[LineItem] = Field(default_factory=list)
    field_confidence: Dict[str, float] = Field(default_factory=dict)
    
class InvoiceState(BaseModel):
    source_path: str
    source_type: Optional[Literal["pdf", "image"]] = None
    images: List[Any] = Field(default_factory=list)
    extracted: Optional[ExtractedInvoice] = None
    status: Literal["pending", "not_invoice", "extracting", "low_confidence", "awaiting_human", "complete", "failed"] = "pending"
```

### 3. `models/loader.py`
**Purpose:** Implements a singleton to load the massive ~14GB VLM model into GPU VRAM only once, preventing slow reloads across agent loops.
**Key Code Snippet:**
```python
def get_model_and_processor():
    global _model, _processor
    if _model is not None and _processor is not None:
        return _model, _processor
    # Loads Qwen2.5-VL once
    _processor = AutoProcessor.from_pretrained(MODEL_ID)
    _model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    return _model, _processor
```

### 4. `utils/image_utils.py`
**Purpose:** Preprocesses scans and PDFs to avoid OOM limits. Includes a router for cropping visual regions when the AI has low confidence.
**Key Code Snippet:**
```python
def crop_region(img: Image.Image, region: str) -> Image.Image:
    w, h = img.size
    if region == "top":       # vendor name, header, invoice number, date
        return img.crop((0, 0, w, int(h * 0.35)))
    elif region == "middle":  # line items
        return img.crop((0, int(h * 0.30), w, int(h * 0.70)))
    elif region == "bottom":  # totals, tax, payment terms
        return img.crop((0, int(h * 0.65), w, h))
```

### 5. `utils/json_utils.py`
**Purpose:** A fuzzy LLM output parser that salvages pure JSON structure even if the AI hallucinates markdown wrappers or trailing commas.
**Key Code Snippet:**
```python
def safe_parse_json(text: str) -> Optional[dict]:
    # Handles markdown wrapping
    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if code_block_match:
        return json.loads(code_block_match.group(1).strip())
    
    # Finds largest curly brace block and fixes trailing commas
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        candidate = re.sub(r",\s*([}\]])", r"\1", brace_match.group(0))
        return json.loads(candidate)
```

### 6. `nodes/input_handler.py`
**Purpose:** The first node of the LangGraph loop. Identifies file types using magic bytes and transforms them safely to clean image sequences.
**Key Code Snippet:**
```python
def input_handler_node(state: InvoiceState) -> InvoiceState:
    try:
        source_type = detect_file_type(state.source_path)
        state.source_type = source_type
        
        if source_type == "pdf":
            state.images = load_pdf_as_images(state.source_path)
        else:
            state.images = load_image(state.source_path)
    except Exception as e:
        state.status = "failed"
        state.error_message = f"Input handler error: {str(e)}"
    return state
```
