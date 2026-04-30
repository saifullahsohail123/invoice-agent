from schema import InvoiceState
from nodes.input_handler import input_handler_node
import os
from PIL import Image

# Create a dummy image for testing
os.makedirs("tests/sample_invoices", exist_ok=True)
img = Image.new('RGB', (800, 600), color = 'white')
test_path = "tests/sample_invoices/dummy.jpg"
img.save(test_path)

state = InvoiceState(source_path=test_path)
state = input_handler_node(state)

if state.status == "failed":
    print("Test failed:", state.error_message)
else:
    print("Test passed! Image loaded.")
    print("Source type:", state.source_type)
    print("Number of images:", len(state.images))
    print("Image size:", state.images[0].size)

# Also test model loading
print("\nTesting model loader...")
from models.loader import get_model_and_processor
model, processor = get_model_and_processor()
print("Model loading test complete.")
