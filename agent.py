from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from schema import InvoiceState
from nodes.input_handler import input_handler_node
from nodes.classifier import classifier_node
from nodes.extractor import extractor_node
from nodes.confidence import confidence_router
from nodes.re_examiner import re_examiner_node
from nodes.human_interrupt import human_interrupt_node
from nodes.storage import storage_node

def build_graph():
    """
    Build and compile the LangGraph invoice extraction agent.
    
    Graph structure:
    
    input_handler
         ↓
    classifier ──[not_invoice]──→ END (with rejection message)
         ↓ [is_invoice]
    extractor
         ↓
    confidence_router ──[high confidence]──→ storage → END
         ↓ [low confidence + retries left]
    re_examiner
         ↓
    confidence_router (loop back)
         ↓ [retries exhausted]
    human_interrupt (LangGraph pauses here for HITL)
         ↓ [after human approval]
    storage → END
    """

    # For Phase 3 testing, we use MemorySaver to maintain state without SQLite complexity
    checkpointer = MemorySaver()

    builder = StateGraph(InvoiceState)

    # Add nodes
    builder.add_node("input_handler", input_handler_node)
    builder.add_node("classifier", classifier_node)
    builder.add_node("extractor", extractor_node)
    builder.add_node("re_examiner", re_examiner_node)
    builder.add_node("human_interrupt", human_interrupt_node)
    builder.add_node("storage", storage_node)

    # Set entry point
    builder.set_entry_point("input_handler")

    # Edges: linear flow
    builder.add_edge("input_handler", "classifier")
    
    # Conditional edge after classifier
    builder.add_conditional_edges(
        "classifier",
        lambda state: "extractor" if getattr(state, "is_invoice", False) else END,
        {
            "extractor": "extractor",
            END: END,
        }
    )

    # Conditional edge after extractor + after re_examiner (same router)
    for source in ["extractor", "re_examiner"]:
        builder.add_conditional_edges(
            source,
            confidence_router,
            {
                "re_examine": "re_examiner",
                "human_interrupt": "human_interrupt",
                "storage": "storage",
                "end_not_invoice": END,
                "end_failed": END,
            }
        )

    # After human approval, route to storage
    builder.add_conditional_edges(
        "human_interrupt",
        confidence_router,
        {
            "storage": "storage",
            "end_failed": END,
        }
    )

    builder.add_edge("storage", END)

    # interrupt_before: pause execution at human_interrupt node
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_interrupt"],
    )

    return graph

def run_invoice(file_path: str, thread_id: str = None) -> InvoiceState:
    import uuid
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = InvoiceState(source_path=file_path)
    final_state = None

    for event in graph.stream(initial_state, config=config):
        final_state = event

    return final_state, thread_id
