import torch
from qwen_vl_utils import process_vision_info
from models.loader import get_model_and_processor
from schema import InvoiceState
from utils.json_utils import safe_parse_json
from utils.image_utils import base64_to_image

CLASSIFY_PROMPT = """
You are a document classification expert.

Examine this document image carefully and determine if it is an invoice, bill, or receipt.

An invoice or receipt MUST contain at least TWO of the following:
1. A vendor/company name or logo
2. A monetary amount or price
3. A date
4. An invoice number, order number, or receipt number
5. Line items, services, or products listed with prices
6. Billing or payment information
7. Words like "Invoice", "Receipt", "Bill", "Statement", "Order"

Respond ONLY with a JSON object in this exact format (no other text):
{
  "is_invoice": true or false,
  "confidence": 0.0 to 1.0,
  "document_type": "invoice" | "receipt" | "bill" | "bank_statement" | "photo" | "screenshot_ui" | "form" | "letter" | "unknown" | "other",
  "reason": "brief explanation in one sentence"
}

Examples of what is NOT an invoice:
- A photo of a person, object, or scene
- A screenshot of a website or app UI
- A blank page or test page
- A letter or memo without billing info
- A product catalog without per-item pricing to a specific buyer
"""

def classifier_node(state: InvoiceState) -> InvoiceState:
    """
    LangGraph node: classifies whether the document is an invoice.
    Uses the first page only (most informative for classification).
    Updates state.is_invoice and state.classification_reason.
    """
    if state.status == "failed" or not state.images:
        return state

    model, processor = get_model_and_processor()
    image = base64_to_image(state.images[0])  # Use first page for classification

    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": CLASSIFY_PROMPT},
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
                max_new_tokens=200,
                temperature=0.1,   # Low temperature for consistent classification
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

        result = safe_parse_json(output_text)

        if result:
            state.is_invoice = result.get("is_invoice", False)
            state.classification_reason = result.get("reason", "Unknown")
            doc_type = result.get("document_type", "unknown")
            
            if not state.is_invoice:
                state.status = "not_invoice"
                state.error_message = (
                    f"This document does not appear to be an invoice or receipt. "
                    f"Detected type: '{doc_type}'. Reason: {state.classification_reason}"
                )
        else:
            # If JSON parsing fails, default to attempting extraction
            # (fail open rather than fail closed)
            state.is_invoice = True
            state.classification_reason = "Classification response could not be parsed — proceeding with extraction."

    except Exception as e:
        # On any model error, fail open
        state.is_invoice = True
        state.classification_reason = f"Classification error (proceeding): {str(e)}"

    return state
