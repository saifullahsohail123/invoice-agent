from schema import InvoiceState

def storage_node(state: InvoiceState) -> InvoiceState:
    """
    LangGraph node: Finalizer node that writes the fully verified Invoice JSON 
    into the SQLite Database. Acts as Stage 2 Duplicate Gate.
    """
    import os
    from db.database import insert_invoice

    if state.status in ("failed", "not_invoice", "duplicate_file", "duplicate_logical"):
        return state

    if not state.extracted:
        return state

    file_name = os.path.basename(state.source_path)
    
    # Store logically while adhering to vendor + invoice composite requirements
    success, message = insert_invoice(
        state.extracted, 
        file_hash=state.file_hash, 
        file_name=file_name
    )

    if success:
        state.status = "complete"
        state.error_message = message # Keep as a positive record
    else:
        state.status = "duplicate_logical"
        state.error_message = message

    return state
