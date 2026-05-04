from agent import run_invoice

def test_full_loop():
    print("Initializing LangGraph Agentic Pipeline...")
    test_path = "tests/sample_invoices/sample.png"
    
    # We execute the graph through its main loop handler
    final_state, thread_id = run_invoice(test_path)
    
    print("\n--- Pipeline Finished ---")
    if getattr(final_state, "value", None):
        # We need to unwrap the state since graph.stream yields dicts representing nodes
        # Actually run_invoice returns the final node output, which is a dict of the state
        # In Pydantic based langgraph, it might return the state object directly or wrapped.
        for node_name, state_obj in final_state.items():
            state = state_obj
    else:
        state = final_state

    # With LangGraph, if it returns a dataclass or a dict
    if isinstance(state, dict):
        state = list(state.values())[0]

    print("Final Status:", state.status)
    if state.status == "complete":
        print(f"Extraction Successful! Overall Confidence: {state.extracted.overall_confidence:.2%}")
        print("\nRetries Used:", state.retry_count)
        print("\n--- Final Extracted Data ---")
        print(state.extracted.model_dump_json(indent=2))
    elif state.status == "awaiting_human":
        print("Agent halted correctly! Escalated to Human-in-the-loop because confidence remained low after retries.")
        print("Low Confidence Fields:", state.extracted.low_confidence_fields)

if __name__ == "__main__":
    test_full_loop()
