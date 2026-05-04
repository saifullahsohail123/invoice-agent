# Invoice Agent Uniqueness & Deduplication Plan

To avoid double-processing the same document and wasting expensive AI compute/VRAM, we implement a **Hybrid Uniqueness Validation** technique.

## 1. Stage One: File Hashing (The Fast Gate)
Before the AI even loads the model or opens the image, we do a rapid binary hash of the entire file.
- **Mechanism:** Read the binary contents of `state.source_path` and generate a `SHA-256` hash.
- **Verification:** The `input_handler` queries a `processed_files` SQLite table. If the hash exists, it instantly sets `state.status = 'duplicate_file'` and LangGraph terminates immediately.
- **Benefit:** 100% compute savings against a user uploading the exact same PDF multiple times.

## 2. Stage Two: Logical Composite Key (The Semantic Gate)
If someone takes two *different* photos of the *same* physical receipt, the binary file hashes will be completely different. Stage One will mistakenly let it through.
- **Mechanism:** The AI processes the image normally and extracts the payload.
- **Verification:** When the graph hits the `storage` node, the node checks the `invoices` SQLite table for an existing row matching the exact combination of `(vendor_name, invoice_number)`.
- **Handling:** If `vendor=SCP Service` and `invoice_number=0000354` already exists, it skips database insertion, flags `state.status = 'duplicate_logical'`, and alerts the user.

## Implementation Steps
This uniqueness implementation is natively woven into **Phase 4**, as both validation stages heavily rely on the SQLite database architecture.

1.  **Architecture:** Build `db/database.py` with SQL schemas containing the explicit `UNIQUE(vendor_name, invoice_number)` constraint and the `processed_files` tracker.
2.  **Input Check:** Update `nodes/input_handler.py` to calculate file hashes and perform the Stage One check.
3.  **Storage Check:** Update `nodes/storage.py` to perform the SQLite Stage Two insertion and duplicate trapping.
4.  **UI Feedback:** Ensure `streamlit_app.py` surfaces these duplicate alerts smoothly to the user.
