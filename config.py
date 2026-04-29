# config.py

from pathlib import Path

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
# Alternative for higher accuracy (needs ~18-19GB VRAM):
# MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct"  # Run via llama.cpp at Q4_K_M
MODEL_DTYPE = "bfloat16"  # Use float16 if bfloat16 causes issues on your GPU
MAX_NEW_TOKENS_EXTRACT = 1500
MAX_NEW_TOKENS_CLASSIFY = 200
MAX_NEW_TOKENS_REEXAMINE = 100
TEMPERATURE = 0.1

# ── Image preprocessing ────────────────────────────────────────────────────────
MAX_IMAGE_LONG_SIDE = 1280   # pixels
PDF_DPI = 200                # DPI for PDF → image conversion
CONTRAST_ENHANCEMENT = 1.2  # slight boost for scanned docs

# ── Agent thresholds ───────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.80  # Fields below this are flagged as low confidence
MAX_RETRIES = 2              # Re-examine attempts before escalating to HITL

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DB_PATH = str(BASE_DIR / "db" / "invoices.db")
EXPORTS_DIR = str(BASE_DIR / "exports")
PROMPTS_DIR = str(BASE_DIR / "prompts")

# ── Ensure directories exist ───────────────────────────────────────────────────
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(EXPORTS_DIR).mkdir(parents=True, exist_ok=True)
Path(PROMPTS_DIR).mkdir(parents=True, exist_ok=True)
