# models/loader.py
# Singleton pattern: model loads once and stays in memory.

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from config import MODEL_ID, MODEL_DTYPE

_model = None
_processor = None

def get_model_and_processor():
    global _model, _processor

    if _model is not None and _processor is not None:
        return _model, _processor

    dtype = torch.bfloat16 if MODEL_DTYPE == "bfloat16" else torch.float16

    print(f"Loading model {MODEL_ID}... (first run downloads ~14GB)")
    _processor = AutoProcessor.from_pretrained(MODEL_ID)
    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",         # Automatically uses your GPU
    )
    _model.eval()
    print("Model loaded successfully.")

    return _model, _processor
