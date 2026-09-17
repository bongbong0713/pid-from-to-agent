"""Smallest end-to-end check of the local VLM: load it, show it one crop, read the JSON back."""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
from src.vlm.tier0 import ocr_words, merge_fragments, locate_label, build_crop, ask_vlm
from src.vlm import local_qwen

TARGET = "16-20001-NPH-250 -W10CB01-HC-50"   # B-0001 -> P-1001/P-1002

img = cv2.imread("data/sample_pid.png")
best = locate_label(merge_fragments(ocr_words(img)), TARGET)[0]
crop, _ = build_crop(img, best["cx"], best["cy"], 1400,
                     (best["x"], best["y"], best["w"], best["h"]))
print(f"crop {crop.shape[1]}x{crop.shape[0]}  vision_tokens={local_qwen.vision_tokens(crop)}")

t = time.time(); local_qwen.load(); print(f"모델 로드 {time.time()-t:.1f}s")
import torch
print(f"VRAM {torch.cuda.memory_allocated()/1e9:.1f}GB / reserved {torch.cuda.memory_reserved()/1e9:.1f}GB")

t = time.time(); r = ask_vlm(crop, TARGET, "local"); dt = time.time() - t
print(f"추론 {dt:.1f}s")
print(json.dumps(r, indent=2, ensure_ascii=False))
print("\n정답: from=B-0001  to=P-1001/P-1002")
