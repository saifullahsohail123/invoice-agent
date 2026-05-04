import streamlit as st
import pandas as pd
import os
import shutil
from agent import build_graph, run_invoice
from schema import InvoiceState
from db.database import get_connection, init_db
from nodes.human_interrupt import apply_human_corrections

# Initialize DB on start
init_db()

st.set_page_config(page_title="Agentic Invoice Extractor", layout="wide")
st.title("🖺 Agentic Invoice Extractor (Qwen2.5-VL)")

# --- Sidebar: File Upload ---
st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader("Upload Invoice (PDF, PNG, JPG)", type=["pdf", "png", "jpg", "jpeg", "webp"])

if uploaded_file is not None:
    # Save the uploaded file locally so the pipeline can process it
    os.makedirs("temp_uploads", exist_ok=True)
    file_path = os.path.join("temp_uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.success(f"Uploaded: {uploaded_file.name}")

    if st.sidebar.button("Process Document"):
        with st.spinner("Agent is processing document..."):
            st.session_state["thread_id"] = uploaded_file.name
            st.session_state["file_path"] = file_path
            
            # Execute pipeline
            final_state, thread_id = run_invoice(file_path, thread_id=st.session_state["thread_id"])
            st.session_state["final_state"] = final_state

# --- Main Area: Interactive Feedback ---
if "final_state" in st.session_state:
    state: InvoiceState = st.session_state["final_state"]
    
    if state.status == "complete":
        st.success("✅ Extraction Complete & Saved to Database!")
        if state.error_message:
            st.info(f"Database Note: {state.error_message}")
        st.json(state.extracted.model_dump())
        
    elif state.status == "duplicate_file":
        st.warning(f"⚠️ Duplicate Blocked: {state.error_message}")
        
    elif state.status == "duplicate_logical":
        st.warning(f"⚠️ Logical Duplicate Blocked: {state.error_message}")
        if state.extracted:
            st.json(state.extracted.model_dump())

    elif state.status == "not_invoice":
        st.error(f"❌ Rejected Document: {state.error_message}")
        
    elif state.status == "failed":
        st.error(f"❌ Execution Failed: {state.error_message}")

    elif state.status == "awaiting_human":
        st.warning(f"⚠️ Human Intervention Required. The agent exhausted its retries.")
        st.info(f"Low Confidence Fields: {', '.join(state.extracted.low_confidence_fields)}")
        
        st.write("### Review & Correct Extracted Data:")
        # Render a UI form strictly for low confidence items
        corrections = {}
        with st.form("human_correction_form"):
            for field in state.extracted.low_confidence_fields:
                current_value = getattr(state.extracted, field, "")
                corrections[field] = st.text_input(f"Correct {field}:", value=str(current_value))
                
            submit = st.form_submit_button("Approve & Save")
            
            if submit:
                # Apply corrections and immediately run the graph to 'storage'
                state = apply_human_corrections(state, corrections)
                
                # We resume the agent graph!
                graph = build_graph()
                config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
                graph.update_state(config, state) # Inject human-updated state
                
                # Step through the remainder of the graph (which is just 'storage')
                final_event = None
                for event in graph.stream(None, config=config):
                    final_event = event
                    
                st.session_state["final_state"] = graph.get_state(config).values
                st.rerun()

# --- Database Viewer Tab ---
st.write("---")
st.subheader("Database Overview")

try:
    conn = get_connection()
    df = pd.read_sql_query("SELECT id, vendor_name, vendor_phone, invoice_number, buyer_name, total, invoice_date, overall_confidence FROM invoices ORDER BY id DESC", conn)
    conn.close()
    
    st.dataframe(df, use_container_width=True)
except Exception as e:
    st.error(f"Could not load database: {e}")
