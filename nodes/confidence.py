from schema import InvoiceState
from config import CONFIDENCE_THRESHOLD, MAX_RETRIES

def confidence_router(state: InvoiceState) -> str:
    """
    LangGraph conditional edge function.
    Returns the name of the next node to route to.
    
    Routing logic:
      - Not an invoice           → "end_not_invoice"
      - Failed state             → "end_failed"
      - No extracted data        → "re_examine" (retry)
      - Low confidence fields
          AND retries remaining  → "re_examine"
          AND retries exhausted  → "human_interrupt"
      - All fields high conf.    → "storage"
      - Human already approved   → "storage"
    """
    if state.status == "not_invoice":
        return "end_not_invoice"

    if state.status == "failed":
        return "end_failed"

    if state.human_approved:
        return "storage"

    if state.extracted is None:
        if state.retry_count < MAX_RETRIES:
            return "re_examine"
        else:
            return "human_interrupt"

    has_low_confidence = len(state.extracted.low_confidence_fields) > 0

    if not has_low_confidence:
        return "storage"

    if state.retry_count < MAX_RETRIES:
        return "re_examine"
    else:
        return "human_interrupt"
