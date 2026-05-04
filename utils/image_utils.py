from PIL import Image, ImageEnhance, ImageFilter
from config import MAX_IMAGE_LONG_SIDE, CONTRAST_ENHANCEMENT
import io
import base64

def image_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    # Save as JPEG for compression and serialization
    img.save(buffered, format="JPEG", quality=95)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def base64_to_image(b64_str: str) -> Image.Image:
    img_data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_data)).convert("RGB")


def preprocess_image(img: Image.Image) -> Image.Image:
    """
    Resize image so the longest side is MAX_IMAGE_LONG_SIDE.
    Lightly enhance contrast for scanned documents.
    Returns a clean RGB PIL Image.
    """
    # Resize keeping aspect ratio
    w, h = img.size
    long_side = max(w, h)
    if long_side > MAX_IMAGE_LONG_SIDE:
        scale = MAX_IMAGE_LONG_SIDE / long_side
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    
    # Light contrast enhancement for scans
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(CONTRAST_ENHANCEMENT)
    
    return img


def crop_region(img: Image.Image, region: str) -> Image.Image:
    """
    Crop a named region of the image for focused re-examination.
    
    Regions:
        "top"    — top 35% (vendor name, header, invoice number, date)
        "middle" — middle 40% (line items)
        "bottom" — bottom 35% (totals, tax, payment terms)
        "full"   — entire image (fallback)
    """
    w, h = img.size
    if region == "top":
        return img.crop((0, 0, w, int(h * 0.35)))
    elif region == "middle":
        return img.crop((0, int(h * 0.30), w, int(h * 0.70)))
    elif region == "bottom":
        return img.crop((0, int(h * 0.65), w, h))
    else:
        return img


# Maps each extractable field to its likely image region
FIELD_TO_REGION = {
    "vendor_name": "top",
    "vendor_address": "top",
    "vendor_phone": "top",
    "invoice_number": "top",
    "invoice_date": "top",
    "due_date": "top",
    "buyer_name": "top",
    "buyer_address": "top",
    "buyer_phone": "top",
    "payment_details": "bottom",
    "line_items": "middle",
    "subtotal": "bottom",
    "tax": "bottom",
    "tax_rate": "bottom",
    "total": "bottom",
    "currency": "bottom",
    "payment_terms": "bottom",
}
