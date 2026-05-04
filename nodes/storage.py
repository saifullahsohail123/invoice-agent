from schema import InvoiceState

def storage_node(state: InvoiceState) -> InvoiceState:
    """
    Placeholder storage node for Phase 3.
    Replaced in Phase 4.
    """
    if str(state.status) != "failed" and str(state.status) != "not_invoice":
        state.status = "complete"
    return state
