# Phase 2: Core Intelligence (Completed)

This phase implements the intelligence engine: passing pre-processed invoice images to the Qwen2.5-VL-7B-Instruct model alongside hardcoded prompts. We successfully implemented the two key ML-heavy nodes of the LangGraph chain.

## Implementation Pipeline

### 1. `nodes/classifier.py`
**Purpose:** Acts as a gatekeeper. Before we spend 1500 tokens repeatedly extracting JSON from an image, this node rapidly determines if the image is actually a bill/invoice/receipt in the first place.
**Mechanics:**
- Submits the first image page with a small max token length (200 tokens).
- Bypasses extraction entirely if it's a blurry photo, a selfie, or a blank page.
- Modifies `state.is_invoice` and logs why it approved or rejected the image.
**Key Code Snippet:**
```python
def classifier_node(state: InvoiceState) -> InvoiceState:
    model, processor = get_model_and_processor()
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": state.images[0]},
            {"type": "text", "text": CLASSIFY_PROMPT},
        ]}
    ]
    # Runs model inference...
    result = safe_parse_json(output_text)
    state.is_invoice = result.get("is_invoice", False)
    if not state.is_invoice:
        state.status = "not_invoice"
```

### 2. `nodes/extractor.py`
**Purpose:** The single heaviest node in the system. Given an image approved by the classifier, it parses and maps visual values directly into our strict JSON schema.
**Mechanics:**
- Feeds the model the `EXTRACT_PROMPT` containing complex schema rules.
- Captures the LLM output and validates it dynamically via `_build_extracted_invoice`.
- Most importantly, flags `low_confidence_fields` when the LLM attaches a confidence score strictly below `< 0.80` to any extracted field.
**Key Code Snippet:**
```python
def extractor_node(state: InvoiceState) -> InvoiceState:
    model, processor = get_model_and_processor()
    # Runs extraction...
    parsed = safe_parse_json(output_text)
    if parsed:
        state.extracted = _build_extracted_invoice(parsed)
    else:
        state.status = "low_confidence"

def _build_extracted_invoice(parsed: dict) -> ExtractedInvoice:
    # Generates a clean Pydantic object
    field_confidence = parsed.get("field_confidence", {})
    low_conf_fields = [f for f, score in field_confidence.items() if score < CONFIDENCE_THRESHOLD]
    # Returns ExtractedInvoice populated with mapped data
```

### 3. `test_phase2.py`
**Purpose:** An integration test script demonstrating how LangGraph states transfer seamlessly through: `Input Handling ➔ Classification ➔ Extraction`. It uses a dummy blank picture to log decisions dynamically.
