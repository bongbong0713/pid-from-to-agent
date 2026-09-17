"""Local Qwen2.5-VL backend, drop-in replacement for tier0.ask_gemini.

Why local matters here: a hosted API downscales the image on its side, which
put a hard ceiling on crop size (measured: 13px label text falls under 8px once
a crop passes ~2548px). Running the model here means we choose `max_pixels`, so
a large crop keeps its text legible and coverage stops fighting readability.
"""

import json
import os
import re
import threading
from typing import Any, Dict, Optional

import cv2
import numpy as np

MODEL_ID = os.getenv("LOCAL_VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")

# 28x28 patches, 2x2 merged -> a 2560px crop costs ~2.1k vision tokens, which is
# affordable and leaves the drawing's small text intact.
MAX_PIXELS = int(os.getenv("LOCAL_VLM_MAX_PIXELS", 2560 * 2560))
MIN_PIXELS = 256 * 28 * 28

_model = None
_proc = None
_lock = threading.Lock()


def load(four_bit: bool = None):
    """Load once and keep it resident; reloading per call would dominate runtime."""
    global _model, _proc
    with _lock:
        if _model is not None:
            return _model, _proc

        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        if four_bit is None:
            four_bit = os.getenv("LOCAL_VLM_4BIT", "1") not in ("0", "false", "False")

        # 4-bit 7B fits one card; "auto" would split across both GPUs and this
        # driver warns that P2P is unreliable on RTX 4000 series.
        # bf16 weights are ~16.5GB and will not fit one 16GB card; split them.
        device_map = os.getenv("LOCAL_VLM_DEVICE", "cuda:0" if four_bit else "auto")
        kwargs: Dict[str, Any] = {"dtype": torch.bfloat16, "device_map": device_map}
        if four_bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, **kwargs)
        _model.eval()
        _proc = AutoProcessor.from_pretrained(
            MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
        )
        return _model, _proc


def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Qwen is not constrained to JSON, so recover the object from the reply."""
    text = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    while start != -1:                      # try progressively shorter prefixes
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                esc = (ch == "\\") and not esc
                if ch == '"' and not esc:
                    in_str = False
            elif ch == '"':
                in_str, esc = True, False
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def ask(crop: np.ndarray, prompt: str, max_new_tokens: int = 512) -> Dict[str, Any]:
    """Run one image+text turn. Same contract as tier0.ask_gemini: dict, or
    {"error": ...} — callers must not have to care which backend answered."""
    try:
        import torch
        from PIL import Image

        model, proc = load()
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        messages = [{"role": "user", "content": [
            {"type": "image", "image": pil},
            {"type": "text", "text": prompt + "\n\nRespond with ONLY the JSON object."},
        ]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[pil], return_tensors="pt").to(model.device)

        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        reply = proc.batch_decode(
            out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )[0]

        parsed = _parse_json(reply)
        if parsed is None:
            return {"error": "could not parse JSON", "raw": reply[:400]}
        return parsed
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def chat(messages: list, max_new_tokens: int = 512) -> Dict[str, Any]:
    """Multi-turn conversation, images and all.

    `messages` is [{"role": "user"|"assistant", "content": [part, ...]}] where a
    part is {"type": "text", "text": ...} or {"type": "image", "image": <BGR array>}.
    The agent loop needs this: each view() the model asks for appends another
    image to the same conversation, so it can compare what it has already seen.
    """
    try:
        import torch
        from PIL import Image

        model, proc = load()
        images, conv = [], []
        for m in messages:
            parts = []
            for c in m["content"]:
                if c["type"] == "image":
                    pil = Image.fromarray(cv2.cvtColor(c["image"], cv2.COLOR_BGR2RGB))
                    images.append(pil)
                    parts.append({"type": "image", "image": pil})
                else:
                    parts.append({"type": "text", "text": c["text"]})
            conv.append({"role": m["role"], "content": parts})

        text = proc.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=images or None, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        reply = proc.batch_decode(
            out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

        parsed = _parse_json(reply)
        if parsed is None:
            return {"error": "could not parse JSON", "raw": reply[:400]}
        return parsed
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def vision_tokens(crop: np.ndarray) -> int:
    """Vision-token cost of a crop under the current max_pixels. Cheap to call,
    and the number that decides how large a crop we can afford."""
    h, w = crop.shape[:2]
    scale = min(1.0, (MAX_PIXELS / (h * w)) ** 0.5)
    return int((h * scale // 28) * (w * scale // 28) / 4)
