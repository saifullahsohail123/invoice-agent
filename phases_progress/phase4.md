# Phase 4: SQLite Database & Human-in-the-Loop Web UI (Completed)

This phase finalizes the application. It incorporates the requested Hybrid Uniqueness Architecture and binds the backend LangGraph engine to a responsive Streamlit Dashboard.

## Key Implementations

### 1. `db/database.py`
**Purpose:** Stores extracted invoices logically and tracks file hashes.
**Mechanics:**
- Imposes strict `UNIQUE(vendor_name, invoice_number)` rules to prevent duplicate logical records.
- Creates a `processed_files` table to track the SHA-256 hashes of all input files.
- Provides `insert_invoice()` matching the robust `ExtractedInvoice` Pydantic payload, mapping objects seamlessly into SQL tables natively.

### 2. Update: `nodes/input_handler.py` (Stage 1 Hardware Validation)
- Incorporates `hashlib.sha256` logic immediately after verifying the document type.
- Checks the `processed_files` SQLite table. If duplicate, flags `duplicate_file` and graph halts directly.

### 3. `nodes/storage.py` (Stage 2 Logical Validation)
**Purpose:** Replaces the dummy block. Connects LangGraph to SQLite.
**Mechanics:** 
- Finalizes the data. Attempts to `insert()` into the DB.
- If it encounters a SQLite Integrity error caused by the `UNIQUE(vendor_name, invoice_number)` constraint, it silently catches it, routes the state to `duplicate_logical`, and stops duplicates.

### 4. `streamlit_app.py`
**Purpose:** The interactive visual interface.
**Mechanics:**
- **Sidebar:** Used for uploading PDFs or images seamlessly via Streamlit caching.
- **Main Flow:** Captures LangGraph Execution states (`is_invoice`, `is_duplicate`).
- **Human In the Loop Form:** Crucially, if `state.status == 'awaiting_human'`, Streamlit dynamically renders `st.text_input` fields ONLY for the explicitly listed low-confidence parameters. Upon submit, resumes LangGraph graph automatically directly into Storage.
- **Database Table:** Automatically fetches and maps `df = pd.read_sql_query...` to show real-time ingestion statistics directly on the UI using `st.dataframe`.
