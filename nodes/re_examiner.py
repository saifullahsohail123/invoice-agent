import torch
from qwen_vl_utils import process_vision_info
from models.loader import get_model_and_processor
from schema import InvoiceState
from utils.image_utils import crop_region, FIELD_TO_REGION, base64_to_image
from utils.json_utils import safe_parse_json

REEXAMINE_PROMPT_TEMPLATE = """
Focus only on extracting the {field_name} from this cropped section of an invoice.

{field_instructions}

Respond ONLY with a JSON object:
{{
  "value": <the extracted value or null>,
  "confidence": <0.0 to 1.0>
}}
"""

FIELD_INSTRUCTIONS = {
    "vendor_name": "What is the company or vendor name? Look for the largest text, logo text, or 'From:' field.",
    "vendor_address": "What is the vendor's address?",
    "vendor_phone": "What is the vendor's phone number?",
    "invoice_number": "What is the invoice number? Look for 'Invoice #', 'INV-', 'Invoice No', 'Reference No'.",
    "invoice_date": "What is the invoice date? Return in YYYY-MM-DD format. Look for 'Date:', 'Invoice Date:'.",
    "due_date": "What is the payment due date? Return in YYYY-MM-DD format. Look for 'Due Date:', 'Pay By:'.",
    "buyer_name": "What is the name of the buyer or billed-to entity?",
    "buyer_address": "What is the address of the buyer?",
    "buyer_phone": "What is the phone number of the buyer?",
    "payment_details": "What are the payment details (e.g. Account number, Routing, Bank info)?",
    "total": "What is the final total amount? Return as a plain number only (e.g. 1240.50). Look for 'Total', 'Amount Due', 'Grand Total'.",
    "subtotal": "What is the subtotal before tax? Return as a plain number only.",
    "tax": "What is the tax amount? Return as a plain number only.",
    "tax_rate": "What is the tax rate percentage? Return as a string like '10%' or '20%'.",
    "currency": "What currency is this invoice in? Return the 3-letter code: USD, GBP, EUR, PKR, etc.",
    "payment_terms": "What are the payment terms? E.g. 'Net 30', 'Due on receipt', etc.",
}

def re_examiner_node(state: InvoiceState) -> InvoiceState:
    """
    LangGraph node: re-examines each low-confidence field individually.
    Crops the image to the relevant region and uses a focused prompt.
    Updates the extracted fields and confidence scores in state.
    """
    if state.status in ("failed", "not_invoice") or not state.images:
        return state

    if state.extracted is None:
        state.retry_count += 1
        return state

    model, processor = get_model_and_processor()
    image = base64_to_image(state.images[state.current_page])
    low_conf_fields = state.extracted.low_confidence_fields.copy()

    for field in low_conf_fields:
        if field not in FIELD_INSTRUCTIONS:
            continue

        region_name = FIELD_TO_REGION.get(field, "full")
        cropped_image = crop_region(image, region_name)
        instructions = FIELD_INSTRUCTIONS[field]
        prompt = REEXAMINE_PROMPT_TEMPLATE.format(
            field_name=field.replace("_", " ").title(),
            field_instructions=instructions,
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": cropped_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        try:
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
                    max_new_tokens=100,
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

            result = safe_parse_json(output_text)
            if result and "value" in result:
                new_value = result["value"]
                new_confidence = result.get("confidence", 0.5)

                # Update the extracted field value
                if hasattr(state.extracted, field) and new_value is not None:
                    setattr(state.extracted, field, new_value)
                    state.extracted.field_confidence[field] = new_confidence

        except Exception:
            # If re-examination fails for a field, leave the original value
            continue

    # Recompute low-confidence fields and overall confidence
    from config import CONFIDENCE_THRESHOLD
    state.extracted.low_confidence_fields = [
        f for f, score in state.extracted.field_confidence.items()
        if score < CONFIDENCE_THRESHOLD
    ]
    scores = list(state.extracted.field_confidence.values())
    state.extracted.overall_confidence = sum(scores) / len(scores) if scores else 0.0
    state.retry_count += 1
    state.status = "extracting"

    return state
