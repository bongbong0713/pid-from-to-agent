import os
import base64
import json
import cv2
import numpy as np
from typing import Optional, Dict, Any

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None

from src.ocr.paddle_ocr import PaddleOCRExtractor
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def _encode_image_to_base64_png(image: np.ndarray) -> str:
    _, buf = cv2.imencode('.png', image)
    b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
    return b64


def vlm_labeler_gemini(crop: np.ndarray, model: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Call Gemini/ChatGoogleGenerativeAI with an image crop and ask for a pipe label.

    Returns dict {"label": str, "confidence": float} or None on error.
    """
    api_key = os.getenv('GOOGLE_API_KEY')
    model_name = model or os.getenv('MODEL_NAME', 'gemini-2.5-flash')
    if not api_key or ChatGoogleGenerativeAI is None:
        logger.info('Gemini VLM not available (missing API key or package)')
        return None

    b64 = _encode_image_to_base64_png(crop)

    llm = ChatGoogleGenerativeAI(model=model_name, api_key=api_key)

    # Construct a compact prompt asking for JSON response
    prompt = (
        'You are given an image of a small region cropped from a P&ID diagram. '
        'Return a JSON object with keys `label` and `confidence`. `label` should '
        'be the pipe identifier text exactly as written (e.g., "300-P-310305-NB01-HC"), '
        '`confidence` should be a float 0.0-1.0. If there is no readable pipe label, '
        'return {"label": "", "confidence": 0.0}. Respond with only the JSON.'
    )

    # Attach image as base64 in the message so model can reason over it if supported
    content = {
        'text': prompt,
        'image_base64': b64,
        'image_format': 'png',
    }

    try:
        # Some deployments accept a single chat call with a text+image payload.
        # The ChatGoogleGenerativeAI wrapper may vary; we use a simple `generate` call.
        res = llm.generate([{"role": "user", "content": json.dumps(content)}])
        text = ''
        # Attempt to extract text from response structure
        if hasattr(res, 'generations'):
            gens = getattr(res, 'generations')
            if gens and isinstance(gens, list):
                # pick first candidate
                first = gens[0]
                if isinstance(first, dict):
                    text = first.get('content', '') or first.get('text', '')
                else:
                    text = str(first)
        else:
            text = str(res)

        # Try to find JSON substring
        jstart = text.find('{')
        if jstart >= 0:
            jtxt = text[jstart:]
            try:
                parsed = json.loads(jtxt)
            except Exception:
                # fallback: try to eval-ish minimal cleanup
                parsed = None
        else:
            parsed = None

        if not parsed:
            logger.debug('VLM response could not be parsed as JSON')
            return None

        label = parsed.get('label', '').strip()
        try:
            confidence = float(parsed.get('confidence', 0.0))
        except Exception:
            confidence = 0.0

        if label == '':
            return {"label": "", "confidence": confidence}

        return {"label": label, "confidence": confidence}

    except Exception as e:
        logger.error(f'VLM labeler error: {e}')
        return None


def vlm_labeler_fallback_ocr(crop: np.ndarray) -> Optional[Dict[str, Any]]:
    """A safe fallback VLM that runs PaddleOCR on the crop and heuristically
    extracts a pipe-like label.

    This is deterministic and does not require external APIs.
    """
    try:
        ocr = PaddleOCRExtractor()
        # PaddleOCRExtractor provides an `extract_from_image` helper that accepts
        # a numpy image; if not, write to a temp file and call extract_text_with_regions.
        regions = ocr.extract_from_image(crop)
    except Exception:
        # Older extractor may not have extract_from_image; fallback to saving to disk
        import tempfile

        _, p = tempfile.mkstemp(suffix='.png')
        cv2.imwrite(p, crop)
        regions = ocr.extract_text_with_regions(p, 0.3)

    # Look for pipe-like tokens in OCR results
    for r in regions:
        txt = r.get('text', '')
        if not txt:
            continue
        norm = ''.join(txt.split()).upper()
        # simple heuristics
        if 'P-' in norm or ('P' in norm and any(ch.isdigit() for ch in norm)):
            return {"label": norm, "confidence": float(r.get('confidence', 0.5))}

    return {"label": "", "confidence": 0.0}


def get_default_vlm_labeler() -> callable:
    """Return a VLM labeler callable: prefer Gemini if available, else fallback OCR."""
    api_key = os.getenv('GOOGLE_API_KEY')
    # Only prefer the real Gemini VLM when explicitly enabled via env var.
    use_real = os.getenv('USE_REAL_VLM', '0') in ('1', 'true', 'True')
    if use_real and api_key and ChatGoogleGenerativeAI is not None:
        def fn(crop: np.ndarray):
            return vlm_labeler_gemini(crop)
        return fn

    # fallback to deterministic OCR-based labeler
    return vlm_labeler_fallback_ocr
