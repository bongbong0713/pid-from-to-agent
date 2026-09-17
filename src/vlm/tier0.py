"""Tier 0 baseline: locate the pipe label, crop around it, ask a VLM for FROM/TO.

No graph, no Hough. One crop, one VLM call. This is the number every fancier
approach has to beat.

Deps: cv2, requests, and the `tesseract` binary. Nothing new to install.
"""

import os
import re
import sys
import json
import base64
import difflib
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
import requests

# Running this file directly (python3 src/vlm/tier0.py) leaves the project root
# off sys.path, so `src.vlm.route` would not import.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# --- OCR -------------------------------------------------------------------

def ocr_words(image: np.ndarray, scale: float = 2.0, min_conf: float = 40) -> List[Dict[str, Any]]:
    """Run tesseract in sparse-text mode, return words in ORIGINAL image coords."""
    big = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    with tempfile.TemporaryDirectory() as td:
        png = str(Path(td) / "in.png")
        cv2.imwrite(png, big)
        subprocess.run(
            ["tesseract", png, str(Path(td) / "out"), "--psm", "11", "tsv"],
            check=True, capture_output=True,
        )
        rows = (Path(td) / "out.tsv").read_text(errors="replace").splitlines()

    words = []
    for row in rows[1:]:
        f = row.split("\t")
        if len(f) < 12 or not f[11].strip():
            continue
        try:
            conf = float(f[10])
        except ValueError:
            continue
        if conf < min_conf:
            continue
        words.append({
            "text": f[11].strip(),
            "conf": conf,
            "x": int(f[6]) / scale, "y": int(f[7]) / scale,
            "w": int(f[8]) / scale, "h": int(f[9]) / scale,
        })
    return words


def merge_fragments(words: List[Dict[str, Any]], gap_ratio: float = 1.2) -> List[Dict[str, Any]]:
    """Glue words that sit on the same baseline into one label.

    P&ID line numbers OCR as several pieces ("10-LCV-110", "10-20000-NPH-250",
    "-W10CB01-HC-50"). Matching a target against the pieces never works, so
    stitch them back together first.
    """
    out = []
    for w in sorted(words, key=lambda w: (round(w["y"] / 10), w["x"])):
        if out:
            p = out[-1]
            same_line = abs((w["y"] + w["h"] / 2) - (p["y"] + p["h"] / 2)) < max(p["h"], w["h"]) * 0.6
            close = 0 <= w["x"] - (p["x"] + p["w"]) < max(p["h"], w["h"]) * gap_ratio * 2
            if same_line and close:
                p["text"] += w["text"]
                p["w"] = w["x"] + w["w"] - p["x"]
                p["h"] = max(p["h"], w["h"])
                p["conf"] = min(p["conf"], w["conf"])
                p["parts"].append(w["text"])
                continue
        out.append({**w, "parts": [w["text"]]})
    return out


def _norm(s: str) -> str:
    """Fold the characters OCR reliably confuses on engineering drawings."""
    s = s.upper()
    s = s.replace("—", "-").replace("–", "-").replace("−", "-")
    s = re.sub(r"\s+", "", s)
    return (s.replace("O", "0").replace("I", "1").replace("L", "1")
             .replace("S", "5").replace("B", "8").replace("Z", "2"))


def locate_label(merged: List[Dict[str, Any]], target: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Rank OCR labels by similarity to the target. OCR misreads digits, so
    never require an exact match here — the VLM does the real reading."""
    t = _norm(target)
    scored = []
    for m in merged:
        c = _norm(m["text"])
        if not c:
            continue
        score = difflib.SequenceMatcher(None, t, c).ratio()
        # Containment promotes a partial read to a near-certain match, but only
        # when the fragment is substantial: every single character of the target
        # is trivially "contained" in it, and on a drawing whose OCR is degraded
        # those one-character scraps otherwise outrank every real candidate.
        if len(c) >= max(6, 0.5 * len(t)) and (t in c or c in t):
            score = max(score, 0.9)
        scored.append({**m, "score": score,
                       "cx": m["x"] + m["w"] / 2, "cy": m["y"] + m["h"] / 2})
    scored.sort(key=lambda d: -d["score"])
    return scored[:top_k]


# --- crop ------------------------------------------------------------------

def build_crop(image: np.ndarray, cx: float, cy: float, size: int,
               target_box: Tuple[int, int, int, int] = None) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Square crop centred on (cx, cy), with a coordinate ruler drawn on it.

    The ruler is not decoration: it is how the model reports where a line
    leaves the crop, which is what a Tier 1 loop would need to step forward.
    Returns (crop, (x0, y0)) so crop coords map back to the original.
    """
    h, w = image.shape[:2]
    half = size // 2
    x0 = int(np.clip(cx - half, 0, max(0, w - size)))
    y0 = int(np.clip(cy - half, 0, max(0, h - size)))
    x1, y1 = min(x0 + size, w), min(y0 + size, h)
    crop = image[y0:y1, x0:x1].copy()

    if target_box is not None:
        bx, by, bw, bh = target_box
        cv2.rectangle(crop, (int(bx - x0) - 6, int(by - y0) - 6),
                      (int(bx - x0 + bw) + 6, int(by - y0 + bh) + 6), (0, 0, 255), 3)

    step = 200
    for gx in range(x0 - x0 % step + step, x1, step):
        cv2.line(crop, (gx - x0, 0), (gx - x0, crop.shape[0]), (200, 200, 200), 1)
        cv2.putText(crop, str(gx), (gx - x0 + 3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
    for gy in range(y0 - y0 % step + step, y1, step):
        cv2.line(crop, (0, gy - y0), (crop.shape[1], gy - y0), (200, 200, 200), 1)
        cv2.putText(crop, str(gy), (3, gy - y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
    return crop, (x0, y0)


def equipment_candidates(merged: List[Dict[str, Any]], x0: int, y0: int,
                         x1: int, y1: int) -> List[str]:
    """Equipment tags visible in the crop, to constrain what the model may answer.

    On a P&ID an equipment tag is set in a larger type than a line number, so
    height separates them without another brittle regex: here the six tags sit
    at 25px against a 13px median, while line numbers stay at 13-14px.
    Unconstrained, the model just names the biggest vessel it can see.
    """
    if not merged:
        return ["Off-page"]
    heights = sorted(m["h"] for m in merged)
    med = heights[len(heights) // 2]
    inside = [m for m in merged
              if x0 <= m["x"] + m["w"] / 2 <= x1 and y0 <= m["y"] + m["h"] / 2 <= y1]

    # Tags are written with or without a separator: B-0001, SC 02, MK 01 A.
    TAG = re.compile(r"[A-Z]{1,3}[- ]?\d{2,5}\s?[A-Z]?$")

    def pick(min_h):
        out = set()
        for m in inside:
            if m["h"] < min_h:
                continue
            t = re.sub(r"\s+", " ", m["text"]).strip().upper()
            if TAG.fullmatch(t):
                out.add(t)
        return out

    # Larger type separates tags from line numbers, but only on drawings that
    # set them differently; where everything is one size that filter yields
    # nothing, and an empty list would force every answer to "Off-page".
    tags = pick(med * 1.4) or pick(0)
    return sorted(tags) + ["Off-page"]


EQ_TYPE_HINTS = {
    "P": "pump", "PM": "pump (motor-driven)", "B": "vessel / tank",
    "W": "heat exchanger", "T": "tower / column", "E": "heat exchanger",
    "D": "drum", "C": "compressor", "F": "filter", "V": "vessel",
}


def equipment_hint(tag: str) -> str:
    """Guess what a tag denotes from its letter prefix.

    Type is what decides flow direction when no arrowhead is legible: a pump
    discharges, a vessel drains from the bottom, steam is a supply.
    """
    m = re.match(r"^([A-Z]{1,3})-", tag.upper())
    return EQ_TYPE_HINTS.get(m.group(1), "equipment") if m else "equipment"


# --- VLM -------------------------------------------------------------------

ROUTE_NOTE = """
The target pipe's own route has been RECOLOURED ORANGE for you - follow only the
orange line, ignore every other black line.
Each RED CIRCLE marks an arrowhead that lies ON that orange route. Read those
arrowheads to get the flow direction; ignore triangles outside the circles
(they are check valves, pump impellers and motor glyphs, not flow arrows).
"""


PROMPT = """You are reading a crop of a P&ID (Piping and Instrumentation Diagram).

Target pipe line number: {target}
It is outlined in RED in this image. Grey gridlines show ORIGINAL drawing pixel coordinates.
{route_note}

{candidates}

Follow the target pipe in BOTH directions from its label and report what it connects.

Rules:
- "from" and "to" MUST each be an equipment tag exactly as printed, or "Off-page".
  NEVER put the pipe line number in "from" or "to" - that is the thing being traced,
  not an endpoint.
- Do not default to the largest vessel. A pump is just as valid an endpoint as a tank;
  check which tag the line physically reaches.
- FROM = where fluid comes from, TO = where it goes. Use flow arrows, pump discharge
  (out), vessel nozzles, and the convention that steam/feed enters and product leaves.
- Report equipment tags exactly as printed.
- Inline valves (V-101, 14-FCV-115) and instruments (PI, TIRC, FC) are NOT endpoints
  - trace straight through them.
- If the pipe leaves the crop or ends at an off-page connector arrow, say "Off-page".
- If you cannot see enough to decide, say "Unknown" - do not guess.
- State the endpoint you name in "reasoning" and make sure it matches "from"/"to".

Return ONLY JSON:
{{"from": "...", "to": "...", "confidence": 0.0-1.0,
  "reasoning": "one or two sentences",
  "leaves_crop": true/false,
  "exit_points": [{{"side": "left|right|top|bottom", "coord": <original px>}}]}}"""


def _candidate_block(candidates: Optional[List[str]]) -> str:
    """Offer the tags found in the crop, but never constrain down to nothing.

    With a real list this narrows the answer space; with only "Off-page" left -
    which happens when a drawing sets tags and line numbers in one type size -
    presenting it as the allowed set would force every answer to "Off-page".
    """
    real = [c for c in (candidates or []) if c != "Off-page"]
    if not real:
        return ('Equipment tags were not extracted for this crop - read them off '
                'the image yourself. "Off-page" is also a valid endpoint.')
    return ("Equipment visible in this crop (these are the ONLY allowed answers):\n"
            + "\n".join(f"  - {c}" for c in real + ["Off-page"]))


def ask_vlm(crop: np.ndarray, target: str, backend: str = None,
            candidates: List[str] = None, route_note: str = None) -> Dict[str, Any]:
    """Route one crop to whichever VLM is configured.

    'local' runs Qwen2.5-VL here, which is what lets a crop stay large: a hosted
    API downscales on its side, and past ~2548px this drawing's 13px labels drop
    below the ~8px the model needs to read them.
    """
    backend = backend or os.getenv("VLM_BACKEND", "local")
    cand = _candidate_block(candidates)
    prompt = PROMPT.format(target=target, candidates=cand, route_note=route_note or "")
    if backend == "local":
        from src.vlm import local_qwen
        return local_qwen.ask(crop, prompt)
    if backend == "gemini":
        return ask_gemini(crop, target, prompt=prompt)
    return {"error": f"unknown backend: {backend}"}


def ask_gemini(crop: np.ndarray, target: str, model: str = None,
               prompt: str = None) -> Dict[str, Any]:
    """Call the Gemini REST API directly. Avoids the langchain/SDK version pins
    in requirements.txt, none of which install on this Python."""
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        return {"error": "GOOGLE_API_KEY not set"}
    model = model or os.getenv("MODEL_NAME", "gemini-2.5-flash")

    ok, buf = cv2.imencode(".png", crop)
    if not ok:
        return {"error": "failed to encode crop"}

    body = {
        "contents": [{"parts": [
            {"text": prompt or target},
            {"inline_data": {"mime_type": "image/png",
                             "data": base64.b64encode(buf.tobytes()).decode()}},
        ]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0},
    }
    try:
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            params={"key": key}, json=body, timeout=120,
        )
        r.raise_for_status()
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text)
    except requests.HTTPError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text[:300]}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# --- orchestration ---------------------------------------------------------

def run(image_path: str, target: str, out_dir: str = "outputs",
        crop_size: int = 1400, call_vlm: bool = True,
        backend: str = None, truth: Dict[str, Any] = None,
        at: Tuple[float, float] = None) -> Dict[str, Any]:
    """Tier 0 end to end, writing a visualization for every stage."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"cannot read image: {image_path}")

    words = ocr_words(image)
    merged = merge_fragments(words)
    cands = locate_label(merged, target)
    if at is not None:
        # Hand-given position. On drawings whose line numbers are 6-8px tall OCR
        # finds almost none of them, and scoring the whole pipeline then only
        # measures tesseract. Pinning the label isolates the crop-and-reason half.
        cx, cy = float(at[0]), float(at[1])
        bw = max(60.0, 7.0 * len(target))
        best = {"text": target, "score": 1.0, "cx": cx, "cy": cy,
                "x": cx - bw / 2, "y": cy - 7.0, "w": bw, "h": 14.0}
        cands = [best] + [c for c in cands if c is not best]
    elif not cands:
        return {"pipe": target, "from": "Unknown", "to": "Unknown",
                "confidence": 0.0, "error": "no OCR text found"}
    else:
        best = cands[0]

    # stage 1: every merged OCR label, with the matched one in red
    v1 = image.copy()
    for m in merged:
        cv2.rectangle(v1, (int(m["x"]), int(m["y"])),
                      (int(m["x"] + m["w"]), int(m["y"] + m["h"])), (160, 160, 160), 2)
    for c in cands[1:]:
        cv2.rectangle(v1, (int(c["x"]), int(c["y"])),
                      (int(c["x"] + c["w"]), int(c["y"] + c["h"])), (0, 165, 255), 3)
    cv2.rectangle(v1, (int(best["x"]) - 4, int(best["y"]) - 4),
                  (int(best["x"] + best["w"]) + 4, int(best["y"] + best["h"]) + 4), (0, 0, 255), 4)
    cv2.putText(v1, f"{best['text']}  score={best['score']:.2f}",
                (int(best["x"]), int(best["y"]) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imwrite(str(out / "01_ocr_candidates.png"), v1)

    from src.vlm import route as _route
    route_info = _route.pipe_route(image, best["cx"], best["cy"])
    arrows = []
    if route_info is not None:
        arrows = _route.arrowheads_on_route(
            image, route_info["mask"],
            exclude=(best["x"], best["y"], best["w"], best["h"]))
        base = _route.render(image, route_info["mask"], arrows)
    else:
        base = image
    crop, (x0, y0) = build_crop(base, best["cx"], best["cy"], crop_size,
                                (best["x"], best["y"], best["w"], best["h"]))

    # stage 2: where that crop sits on the full drawing
    v2 = image.copy()
    cv2.rectangle(v2, (x0, y0), (x0 + crop.shape[1], y0 + crop.shape[0]), (255, 0, 255), 5)
    cv2.imwrite(str(out / "02_crop_region.png"), v2)

    # stage 3: the exact bytes the VLM sees
    cv2.imwrite(str(out / "03_vlm_input.png"), crop)

    result = {"pipe": target, "from": "Unknown", "to": "Unknown", "confidence": 0.0}
    if call_vlm:
        cands = equipment_candidates(merged, x0, y0,
                                     x0 + crop.shape[1], y0 + crop.shape[0])
        vlm = ask_vlm(crop, target, backend, cands,
                      ROUTE_NOTE if route_info is not None else None)
        result["candidates"] = cands
        if "error" in vlm:
            result["error"] = vlm["error"]
        else:
            result.update({k: vlm.get(k) for k in
                           ("from", "to", "confidence", "reasoning", "leaves_crop")})
    result.update({"source": f"tier0_{backend or os.getenv('VLM_BACKEND', 'local')}",
                   "route_found": route_info is not None,
                   "route_area": route_info["area"] if route_info else 0,
                   "arrows_on_route": len(arrows),
                   "matched_text": best["text"], "match_score": round(best["score"], 3),
                   "label_xy": [round(best["cx"], 1), round(best["cy"], 1)],
                   "crop_origin": [x0, y0], "crop_size": crop_size})

    # stage 4: prediction AND ground truth on the crop, so a glance says pass/fail
    v4 = crop.copy()
    GREEN, RED, GREY = (0, 140, 0), (0, 0, 220), (90, 90, 90)

    def _ok(pred, accepted):
        n = lambda x: str(x or "").strip().upper().replace(" ", "")
        return any(n(pred) == n(a) for a in (accepted or []))

    band = np.full((210 if truth else 150, v4.shape[1], 3), 255, np.uint8)
    F = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(band, target, (12, 32), F, 0.75, (0, 0, 0), 2)

    if truth:
        ok_f, ok_t = _ok(result.get("from"), truth.get("from")), _ok(result.get("to"), truth.get("to"))
        cv2.putText(band, f"PRED  FROM: {result.get('from')}", (12, 74), F, 0.8,
                    GREEN if ok_f else RED, 2)
        cv2.putText(band, f"TO: {result.get('to')}", (v4.shape[1] // 2, 74), F, 0.8,
                    GREEN if ok_t else RED, 2)
        cv2.putText(band, f"GT    FROM: {' / '.join(truth.get('from', []))}", (12, 112), F, 0.7, GREY, 2)
        cv2.putText(band, f"TO: {' / '.join(truth.get('to', []))}", (v4.shape[1] // 2, 112), F, 0.7, GREY, 2)
        verdict = "PASS (both)" if ok_f and ok_t else ("PARTIAL" if ok_f or ok_t else "FAIL")
        cv2.putText(band, verdict, (12, 156), F, 1.0,
                    GREEN if ok_f and ok_t else RED, 3)
        result["ok_from"], result["ok_to"] = ok_f, ok_t
        cv2.putText(band, f"conf={result.get('confidence')}  {result.get('error','')}"[:110],
                    (12, 192), F, 0.55, GREY, 1)
    else:
        cv2.putText(band, f"FROM: {result.get('from')}    TO: {result.get('to')}",
                    (12, 78), F, 0.9, GREEN, 2)
        cv2.putText(band, f"conf={result.get('confidence')}  {result.get('error','')}"[:110],
                    (12, 118), F, 0.6, GREY, 2)
    cv2.imwrite(str(out / "04_result.png"), np.vstack([band, v4]))

    (out / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _self_check():
    """Fragment merge + fuzzy match are the only non-trivial logic here."""
    words = [  # real tesseract output for one label, split into three pieces
        {"text": "10-LCV-110",       "conf": 87, "x": 139, "y": 99, "w": 90, "h": 12},
        {"text": "10-20000-NPH-250", "conf": 61, "x": 229, "y": 99, "w": 145, "h": 12},
        {"text": "-W10CB01-HC-50",   "conf": 83, "x": 374, "y": 99, "w": 130, "h": 12},
        {"text": "PM-1001",          "conf": 88, "x": 988, "y": 1630, "w": 80, "h": 14},
    ]
    merged = merge_fragments(words)
    assert len(merged) == 2, merged
    assert merged[0]["text"] == "10-LCV-11010-20000-NPH-250-W10CB01-HC-50", merged[0]["text"]

    # exact target vs OCR'd digits: must still rank first
    hits = locate_label(merged, "10-LCV-11010-20000-NPH-250 -W10CB01-HC-50")
    assert hits[0]["text"].startswith("10-LCV"), hits[0]
    assert hits[0]["score"] > 0.9, hits[0]["score"]

    # tags written without a dash must be recognised, and an empty tag list must
    # never be presented as the allowed answer set
    tagged = merge_fragments([
        {"text": "SC 02",         "conf": 90, "x": 0, "y": 0,  "w": 40, "h": 9},
        {"text": "MK 01 A",       "conf": 90, "x": 0, "y": 30, "w": 50, "h": 9},
        {"text": "150-Wr-013-NI", "conf": 90, "x": 0, "y": 60, "w": 90, "h": 9},
    ])
    got = equipment_candidates(tagged, -10, -10, 500, 500)
    assert "SC 02" in got and "MK 01 A" in got, got
    assert not any("WR" in g.upper() for g in got), got
    assert "ONLY allowed" in _candidate_block(["SC 02", "Off-page"])
    assert "ONLY allowed" not in _candidate_block(["Off-page"])
    assert "ONLY allowed" not in _candidate_block([])

    # a hand-given position must win regardless of what OCR saw
    import tempfile as _tf, os as _os
    _img = np.full((400, 400, 3), 255, np.uint8)
    cv2.putText(_img, "X-1", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    _p = _os.path.join(_tf.mkdtemp(), "t.png")
    cv2.imwrite(_p, _img)
    _r = run(_p, "77-Zz-999-QQ", _os.path.join(_tf.mkdtemp(), "o"),
             crop_size=200, call_vlm=False, at=(150, 250))
    assert _r["match_score"] == 1.0 and _r["label_xy"] == [150.0, 250.0], _r
    assert _r["crop_origin"] == [50, 150], _r["crop_origin"]

    # OCR digit damage ("19"->"18") must not break the match
    damaged = merge_fragments([{"text": "18-30001-NPH-200", "conf": 86, "x": 0, "y": 0, "w": 9, "h": 9}])
    assert locate_label(damaged, "19-30001-NPH-200")[0]["score"] > 0.9

    # a genuinely different label must NOT outrank the right one
    assert locate_label(merged, "PM-1001")[0]["text"] == "PM-1001"

    # short scraps must not win on containment alone: "T" sits inside
    # "80-ST-001-HC", and on a poorly OCR'd drawing such scraps are everywhere
    scraps = merge_fragments([
        {"text": "T",            "conf": 90, "x": 10, "y": 10, "w": 5,  "h": 9},
        {"text": "80-St-001-HC", "conf": 40, "x": 10, "y": 90, "w": 70, "h": 9},
    ])
    top = locate_label(scraps, "80-St-001-HC")[0]
    assert top["text"] == "80-St-001-HC", top
    assert locate_label(scraps, "150-Wr-013-NI")[0]["text"] != "T"
    print("self-check OK")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Tier 0 VLM baseline for P&ID FROM-TO")
    p.add_argument("image", nargs="?"); p.add_argument("target", nargs="?")
    p.add_argument("--out", default="outputs"); p.add_argument("--crop-size", type=int, default=1400)
    p.add_argument("--no-vlm", action="store_true", help="crops + visualizations only, no API call")
    p.add_argument("--backend", default=None, choices=["local", "gemini"])
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    if a.self_check:
        _self_check()
    else:
        print(json.dumps(run(a.image, a.target, a.out, a.crop_size, not a.no_vlm, a.backend),
                         indent=2, ensure_ascii=False))
