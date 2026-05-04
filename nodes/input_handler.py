# nodes/input_handler.py

import filetype
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance
from pathlib import Path
from schema import InvoiceState
from utils.image_utils import preprocess_image, image_to_base64

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp", ".bmp"}
SUPPORTED_PDF_EXTENSIONS = {".pdf"}

def detect_file_type(path: str) -> str:
    """
    Use the filetype library to detect by magic bytes (not extension).
    Returns 'pdf', 'image', or raises ValueError.
    Crucially: a screenshot saved as .png is detected as 'image'.
    """
    kind = filetype.guess(path)
    if kind is None:
        # Fallback to extension
        ext = Path(path).suffix.lower()
        if ext in SUPPORTED_PDF_EXTENSIONS:
            return "pdf"
        elif ext in SUPPORTED_IMAGE_EXTENSIONS:
            return "image"
        else:
            raise ValueError(f"Unsupported file type: {path}")
    
    if kind.mime == "application/pdf":
        return "pdf"
    elif kind.mime.startswith("image/"):
        return "image"
    else:
        raise ValueError(f"Unsupported MIME type: {kind.mime}")


def load_pdf_as_images(path: str, dpi: int = 200) -> list:
    """
    Convert each PDF page to a PIL Image at the given DPI.
    Use 200 DPI as default — good balance of quality and VRAM usage.
    Falls back to PyMuPDF if pdf2image fails (e.g. encrypted PDFs).
    """
    try:
        images = convert_from_path(path, dpi=dpi, fmt="RGB")
        return [image_to_base64(preprocess_image(img)) for img in images]
    except Exception:
        # Fallback: PyMuPDF
        doc = fitz.open(path)
        images = []
        for page in doc:
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(image_to_base64(preprocess_image(img)))
        return images


def load_image(path: str) -> list:
    """
    Load a single image file. Returns a list with one base64 encoded PIL Image string.
    Handles EXIF rotation, composites transparent images over a white background,
    and converts to RGB.
    """
    img = Image.open(path)
    # Handle EXIF orientation
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    
    # Handle transparency by creating a white background
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        alpha = img.convert('RGBA').split()[-1]
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=alpha)
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
        
    return [image_to_base64(preprocess_image(img))]


def input_handler_node(state: InvoiceState) -> InvoiceState:
    """
    LangGraph node: loads the source file into a list of PIL Images.
    Updates state.images and state.source_type.
    """
    try:
        source_type = detect_file_type(state.source_path)
        state.source_type = source_type

        if source_type == "pdf":
            state.images = load_pdf_as_images(state.source_path)
        else:
            state.images = load_image(state.source_path)

        if not state.images:
            raise ValueError("No images could be loaded from the file.")

    except Exception as e:
        state.status = "failed"
        state.error_message = f"Input handler error: {str(e)}"

    return state
