from schema import InvoiceState
from nodes.input_handler import input_handler_node
from nodes.classifier import classifier_node
from nodes.extractor import extractor_node
import os
from PIL import Image

# Dummy prompt, we won't test extraction on a fully blank image as it might hallucinate text
# But we can test the pipeline stages

def test_pipeline():
    # os.makedirs("tests/sample_invoices", exist_ok=True)
    # img = Image.new('RGB', (800, 600), color='white')
    test_path = "tests/sample_invoices/sample.png"
    # img.save(test_path)

    # 1. Input Handler Node
    state = InvoiceState(source_path=test_path)
    state = input_handler_node(state)
    print("After input handler - images loaded:", len(state.images))

    # 2. Classifier Node
    print("\nRunning Classifier (this will take a moment)...")
    state = classifier_node(state)
    print("Classified as Invoice?", state.is_invoice)
    print("Classification Reason:", state.classification_reason)
    
    if state.status == "not_invoice":
        print("Pipeline correctly stopped at classification.")
    else:
        # Note: on a blank image, it might fail open or say False based on our prompt.
        print("\nRunning Extractor...")
        state = extractor_node(state)
        print("Extractor Status:", state.status)
        if state.extracted:
            print("Overall Confidence:", state.extracted.overall_confidence)

if __name__ == "__main__":
    test_pipeline()
