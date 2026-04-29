# Invoice Agent Implementation Phases

This document outlines the step-by-step implementation plan for the Invoice & Receipt Extraction Agent.

## Phase 1: Foundation & Environment (The Skeleton)
**Goal:** Setup the project structure, dependencies, and core utility modules.
- [ ] Initialize repository and folder structure.
- [ ] Create `requirements.txt` and setup virtual environment.
- [ ] Implement `config.py` (Constants, paths, thresholds).
- [ ] Implement `schema.py` (Pydantic models for State, Invoice, and DB).
- [ ] Implement `models/loader.py` (Singleton pattern for Qwen-VL model loading).
- [ ] Implement `utils/image_utils.py` (Preprocessing, resizing, and region cropping).
- [ ] Implement `utils/json_utils.py` (Robust JSON extraction from LLM text).
- [ ] Implement `nodes/input_handler.py` (PDF/Image to PIL conversion).
- [ ] **Milestone:** Successfully load the model and convert a test PDF into preprocessed images.

## Phase 2: Core Intelligence (The Brain)
**Goal:** Implement the primary VLM-powered nodes for document understanding.
- [ ] Implement `nodes/classifier.py` (Invoice vs. Non-invoice logic).
- [ ] Implement `nodes/extractor.py` (Full field extraction logic).
- [ ] Create prompt templates in `prompts/` (Classification & Extraction).
- [ ] **Testing:** Verify classification accuracy and extraction quality on sample invoices.
- [ ] **Milestone:** Feed an image to the system and receive a structured JSON extraction.

## Phase 3: Agentic Logic & Self-Correction (The Agent)
**Goal:** Build the LangGraph state machine and the "Re-examination" loop.
- [ ] Implement `nodes/confidence.py` (Routing logic based on confidence scores).
- [ ] Implement `nodes/re_examiner.py` (Focused cropping and single-field re-check).
- [ ] Implement `nodes/human_interrupt.py` (HITL state flagging).
- [ ] Build the full graph in `agent.py` using `StateGraph`.
- [ ] **Milestone:** A working agent loop that automatically retries low-confidence fields by "zooming in" on them.

## Phase 4: Interface & Storage (The Product)
**Goal:** Connect the agent to a database and build the user-facing interfaces.
- [ ] Implement `nodes/storage.py` (SQLite persistence and CSV export).
- [ ] Create `main.py` (CLI tool for batch processing).
- [ ] Create `ui/streamlit_app.py` (Dashboard for review, HITL, and database view).
- [ ] **Testing:** End-to-end integration testing (Upload -> Agent -> HITL -> DB).
- [ ] **Milestone:** A complete, usable application with both CLI and Web UI.
