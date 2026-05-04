import torch
from qwen_vl_utils import process_vision_info
from models.loader import get_model_and_processor
from schema import InvoiceState, ExtractedInvoice, LineItem
from utils.json_utils import safe_parse_json
from config import CONFIDENCE_THRESHOLD

EXTRACT_PROMPT = """
You are an expert invoice data extraction system.

Extract ALL information from this invoice image into a structured JSON object.

For each field, you MUST also provide a confidence score between 0.0 and 1.0:
- 1.0 = you can clearly read this value with complete certainty
- 0.7-0.9 = you can read it but some characters may be slightly unclear
- 0.5-0.7 = the value is partially readable or inferred
- below 0.5 = you cannot confidently determine this value

Respond ONLY with a valid JSON object in this exact format (no markdown, no extra text):

{
  "vendor_name": "string or null",
  "vendor_address": "string or null",
  "vendor_phone": "string or null",
  "invoice_number": "string or null",
  "invoice_date": "YYYY-MM-DD or null",
  "due_date": "YYYY-MM-DD or null",
  "buyer_name": "string or null",
  "buyer_address": "string or null",
  "buyer_phone": "string or null",
  "payment_details": "string or null (e.g. Account number, routing, bank details)",
  "line_items": [
    {
      "description": "string",
      "quantity": number or null,
      "unit_price": number or null,
      "total": number or null,
      "confidence": 0.0-1.0
    }
  ],
  "subtotal": number or null,
  "tax": number or null,
  "tax_rate": "string or null (e.g. '10%' or '20%')",
  "total": number or null,
  "currency": "3-letter code e.g. USD, GBP, EUR, PKR",
  "payment_terms": "string or null",
  "notes": "string or null",
  "field_confidence": {
    "vendor_name": 0.0-1.0,
    "vendor_address": 0.0-1.0,
    "vendor_phone": 0.0-1.0,
    "invoice_number": 0.0-1.0,
    "invoice_date": 0.0-1.0,
    "due_date": 0.0-1.0,
    "buyer_name": 0.0-1.0,
    "buyer_address": 0.0-1.0,
    "buyer_phone": 0.0-1.0,
    "payment_details": 0.0-1.0,
    "subtotal": 0.0-1.0,
    "tax": 0.0-1.0,
    "total": 0.0-1.0,
    "currency": 0.0-1.0,
    "payment_terms": 0.0-1.0
  }
}

IMPORTANT:
- Dates MUST be in YYYY-MM-DD format. If you see "15 Apr 2025" convert it to "2025-04-15".
- Monetary values MUST be plain numbers (no currency symbols). E.g. 1240.50 not "$1,240.50".
- If a field is not present in the invoice, use null.
- For currency, detect from symbols: $ = USD, £ = GBP, € = EUR, ₨ or Rs = PKR, ¥ = JPY, etc.
"""


def extractor_node(state: InvoiceState) -> InvoiceState:
    """
    LangGraph node: runs Qwen-VL extraction on the invoice image(s).
    For multi-page PDFs, processes the first page by default.
    (Multi-page handling can be extended later.)
    """
    if state.status in ("failed", "not_invoice") or not state.images:
        return state

    model, processor = get_model_and_processor()
    state.status = "extracting"

    try:
        # For multi-page PDFs, use page index (default 0)
        image = state.images[state.current_page]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": EXTRACT_PROMPT},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1500,
                temperature=0.1,
                do_sample=True,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        state.raw_llm_output = output_text
        parsed = safe_parse_json(output_text)

        if parsed:
            state.extracted = _build_extracted_invoice(parsed)
        else:
            state.retry_count += 1
            state.status = "low_confidence"
            state.error_message = "Could not parse extraction output as JSON."

    except Exception as e:
        state.status = "failed"
        state.error_message = f"Extraction error: {str(e)}"

    return state


def _build_extracted_invoice(parsed: dict) -> ExtractedInvoice:
    """Build an ExtractedInvoice from the raw parsed dict and compute overall confidence."""
    line_items = []
    for item in parsed.get("line_items", []):
        line_items.append(LineItem(
            description=item.get("description", ""),
            quantity=item.get("quantity"),
            unit_price=item.get("unit_price"),
            total=item.get("total"),
            confidence=item.get("confidence", 1.0),
        ))

    field_confidence = parsed.get("field_confidence", {})
    low_conf_fields = [
        field for field, score in field_confidence.items()
        if score < CONFIDENCE_THRESHOLD
    ]

    confidences = list(field_confidence.values())
    overall = sum(confidences) / len(confidences) if confidences else 0.0

    return ExtractedInvoice(
        vendor_name=parsed.get("vendor_name"),
        vendor_address=parsed.get("vendor_address"),
        vendor_phone=parsed.get("vendor_phone"),
        invoice_number=parsed.get("invoice_number"),
        invoice_date=parsed.get("invoice_date"),
        due_date=parsed.get("due_date"),
        buyer_name=parsed.get("buyer_name"),
        buyer_address=parsed.get("buyer_address"),
        buyer_phone=parsed.get("buyer_phone"),
        payment_details=parsed.get("payment_details"),
        line_items=line_items,
        subtotal=parsed.get("subtotal"),
        tax=parsed.get("tax"),
        tax_rate=parsed.get("tax_rate"),
        total=parsed.get("total"),
        currency=parsed.get("currency", "USD"),
        payment_terms=parsed.get("payment_terms"),
        notes=parsed.get("notes"),
        field_confidence=field_confidence,
        low_confidence_fields=low_conf_fields,
        overall_confidence=overall,
    )
