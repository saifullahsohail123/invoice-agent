# utils/json_utils.py
# The VLM sometimes wraps JSON in markdown code blocks or adds extra text.
# This utility robustly extracts JSON from messy LLM outputs.

import json
import re
from typing import Optional

def safe_parse_json(text: str) -> Optional[dict]:
    """
    Attempt to parse JSON from a VLM output string.
    
    Handles:
    1. Clean JSON output
    2. JSON wrapped in ```json ... ``` markdown
    3. JSON embedded in prose ("Here is the data: {...}")
    4. Single quotes instead of double quotes
    5. Trailing commas (common LLM mistake)
    """
    if not text or not text.strip():
        return None

    # Strategy 1: Direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from markdown code block
    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 3: Find the largest {...} block in the text
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        candidate = brace_match.group(0)
        # Fix trailing commas before } or ]
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Strategy 4: Replace single quotes (only for simple cases)
    try:
        fixed = text.strip().replace("'", '"')
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    return None
