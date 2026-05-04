# Phase 3: Agentic Logic & Self-Correction (Completed)

This phase elevates the static extraction pipeline into a fully autonomous agent using LangGraph. The agent can evaluate its own performance and, if it detects ambiguity, physically "squint" closer at the document to try and correct itself before asking a human for help.

## Updates to Phase 2 Based on Business Needs
Due to missed fields on our test invoice, we updated `schema.py`, `nodes/extractor.py`, and `utils/image_utils.py` to capture extra vital fields:
*   `buyer_name`, `buyer_address`, `buyer_phone`
*   `payment_details`
*   `vendor_phone`

## Implementation Pipeline

### 1. `nodes/confidence.py`
**Purpose:** The central logic crossroad of the agent loop.
**Mechanics:**
- It is a LangGraph conditional router. After the `extractor` or `re_examiner` finishes, this node looks at the confidence values returned by the model.
- If everything is `≥ 0.80`, it passes execution to `storage`.
- If fields are `< 0.80` and `retry_count < MAX_RETRIES`, it routes to the `re_examiner`.
- If fields are `< 0.80` but retries are exhausted, it routes to `human_interrupt`.

### 2. `nodes/re_examiner.py`
**Purpose:** The "Squinting" Agent. An advanced self-correction module.
**Mechanics:**
- Instead of forcing the LLM to read the entire massive image again (which costs tokens and didn't work the first time), it crops the image dynamically based on `FIELD_TO_REGION`.
- Example: If `payment_details` has a confidence of 0.4, it crops the bottom 35% of the invoice and feeds *only that sliver* to the LLM with a laser-focused prompt: *"What are the payment details?"*
- Overwrites the main JSON extraction with the new, higher-confidence values.

### 3. `nodes/human_interrupt.py`
**Purpose:** LangGraph interrupt point.
**Mechanics:**
- If the agent fails, we don't want to crash. We gracefully flag `state.awaiting_human = True`.
- LangGraph pauses graph execution here. Later in Streamlit, a human can type in the missed value, which unlocks the graph and continues to storage.

### 4. `agent.py`
**Purpose:** The architecture blueprint.
**Mechanics:**
- Combines all individual nodes into a `StateGraph`.
- Defines exactly how states flow using `add_conditional_edges`.
- Uses `MemorySaver` to checkpoint progress. Crucial for resuming after a human corrects the data!

### 5. `test_phase3.py`
**Purpose:** Integration script for the Agentic loop. Executes the graph using `run_invoice(filepath)` and tracks how many retries the LangGraph system utilized before halting or succeeding.
