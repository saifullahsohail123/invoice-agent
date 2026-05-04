from schema import InvoiceState

def human_interrupt_node(state: InvoiceState) -> InvoiceState:
    """
    LangGraph node: flags the state as awaiting human review.
    The actual UI interaction happens in Streamlit (streamlit_app.py).
    This node sets the flag; LangGraph's interrupt_before mechanism
    pauses graph execution here.
    """
    state.awaiting_human = True
    state.status = "awaiting_human"
    return state

def apply_human_corrections(state: InvoiceState, corrections: dict) -> InvoiceState:
    """
    Called by the Streamlit UI after the human submits corrections.
    Applies corrections to extracted fields and marks as approved.
    
    corrections: dict mapping field_name → corrected_value
    e.g. {"total": 1240.50, "invoice_date": "2025-04-15"}
    """
    if state.extracted is None:
        from schema import ExtractedInvoice
        state.extracted = ExtractedInvoice()

    for field, value in corrections.items():
        if hasattr(state.extracted, field):
            setattr(state.extracted, field, value)
            # Human-corrected fields get maximum confidence
            state.extracted.field_confidence[field] = 1.0

    # Recompute low-confidence fields after corrections
    from config import CONFIDENCE_THRESHOLD
    state.extracted.low_confidence_fields = [
        f for f, score in state.extracted.field_confidence.items()
        if score < CONFIDENCE_THRESHOLD
    ]

    state.awaiting_human = False
    state.human_approved = True
    state.status = "complete"
    return state
