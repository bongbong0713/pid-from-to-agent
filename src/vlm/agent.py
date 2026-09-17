"""Agent loop: the same pipeline, but the VLM decides which step to run next.

tier0.py runs OCR -> locate -> route -> crop -> ask in a fixed order and calls the
model once. Here each of those becomes a tool the model invokes itself, so it can
look somewhere, decide the view was wrong, and look again.

Measured caveat, so nobody expects magic: on the 2201px sample drawing this does
not beat tier0, because every failure there had both endpoints already inside the
single crop. It earns its keep when one crop cannot cover the drawing - a real
scan at 5000-10000px against a ~2560px readable crop ceiling.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.vlm import route as _route
from src.vlm.tier0 import (build_crop, equipment_candidates, locate_label,
                           merge_fragments, ocr_words)

MAX_VIEWS = 6          # each view costs ~625 vision tokens and they accumulate


class Toolbox:
    """The pipeline stages, exposed as callables with a shared cache.

    OCR is the expensive step (~4s), so it runs once and every tool reads the
    same word list.
    """

    def __init__(self, image: np.ndarray, target: str, max_view: int = 2560):
        self.image = image
        self.target = target
        # Caps how much the model may see at once. Lowering it simulates a larger
        # drawing: what decides whether one view suffices is the crop-to-drawing
        # ratio, not the absolute pixel count.
        self.max_view = int(max_view)
        self.h, self.w = image.shape[:2]
        self._merged = None
        self._routes: Dict[Tuple[int, int], Any] = {}
        self.views: List[Dict[str, Any]] = []

    @property
    def merged(self):
        if self._merged is None:
            self._merged = merge_fragments(ocr_words(self.image))
        return self._merged

    # --- tools -------------------------------------------------------------

    def find_label(self, text: str = None, top_k: int = 5) -> Dict[str, Any]:
        """Where a line number or tag sits. Fuzzy - OCR damages digits."""
        hits = locate_label(self.merged, text or self.target, top_k)
        return {"matches": [
            {"text": h["text"], "x": round(h["cx"]), "y": round(h["cy"]),
             "score": round(h["score"], 3)} for h in hits]}

    def list_equipment(self) -> Dict[str, Any]:
        """Equipment tags on the drawing, told apart from line numbers by type size."""
        tags = equipment_candidates(self.merged, 0, 0, self.w, self.h)
        out = []
        for t in tags:
            if t == "Off-page":
                continue
            h = locate_label(self.merged, t, 1)
            if h:
                out.append({"tag": t, "x": round(h[0]["cx"]), "y": round(h[0]["cy"])})
        return {"equipment": out, "note": "'Off-page' is also a valid endpoint"}

    def trace_route(self, x: float, y: float) -> Dict[str, Any]:
        """Follow the piping run passing through (x, y).

        Uses stroke thickness, not Hough: process piping is drawn heavier than
        instrument lines. Thin service lines can fall below the threshold, in
        which case this reports found=False rather than guessing.
        """
        key = (int(x) // 20, int(y) // 20)
        if key not in self._routes:
            self._routes[key] = _route.pipe_route(self.image, x, y)
        r = self._routes[key]
        if r is None:
            return {"found": False,
                    "reason": "this line is too thin to isolate from the drawing "
                              "(thin service lines are drawn at instrument weight)",
                    "next": "call nearest_equipment(x, y) and view() toward the "
                            "candidates instead - do NOT re-view the same spot"}
        bx, by, bw, bh = r["bbox"]
        arrows = _route.arrowheads_on_route(self.image, r["mask"])
        return {"found": True,
                "extent": {"x_min": bx, "y_min": by, "x_max": bx + bw, "y_max": by + bh},
                "arrowheads": [{"x": int(ax), "y": int(ay)} for ax, ay in arrows],
                "hint": "view() the ends of this extent; the route is drawn orange "
                        "and arrowheads are ringed red"}


    def nearest_equipment(self, x: float, y: float, k: int = 4) -> Dict[str, Any]:
        """Equipment nearest a point, with a bearing to each.

        Route extraction fails on thin service lines, and without it the model
        had nowhere to go and re-viewed the same spot until its budget ran out.
        This does not need a route: it names directions worth looking in.
        """
        out = []
        for t in equipment_candidates(self.merged, 0, 0, self.w, self.h):
            if t == "Off-page":
                continue
            h = locate_label(self.merged, t, 1)
            if not h:
                continue
            dx, dy = h[0]["cx"] - x, h[0]["cy"] - y
            bearing = ("right" if dx > 0 else "left") if abs(dx) > abs(dy) else \
                      ("down" if dy > 0 else "up")
            out.append({"tag": t, "x": round(h[0]["cx"]), "y": round(h[0]["cy"]),
                        "distance": round((dx * dx + dy * dy) ** 0.5),
                        "direction": bearing})
        out.sort(key=lambda d: d["distance"])
        return {"nearest": out[:int(k)],
                "hint": "view() midway between here and a candidate to see whether "
                        "the pipe actually runs that way"}

    def view(self, x: float, y: float, size: int = 1400,
             highlight_route: bool = True) -> Dict[str, Any]:
        """Look at the drawing around (x, y). Returns an image to the model."""
        size = int(max(200, min(size, self.max_view)))   # below: unreadable
        base = self.image
        route_on = False
        if highlight_route:
            r = self._routes.get((int(x) // 20, int(y) // 20)) or \
                _route.pipe_route(self.image, x, y)
            if r is not None:
                lab = locate_label(self.merged, self.target, 1)
                ex = (lab[0]["x"], lab[0]["y"], lab[0]["w"], lab[0]["h"]) if lab else None
                arrows = _route.arrowheads_on_route(self.image, r["mask"], exclude=ex)
                base = _route.render(self.image, r["mask"], arrows)
                route_on = True
        crop, (x0, y0) = build_crop(base, x, y, size)
        box = (x0, y0, x0 + crop.shape[1], y0 + crop.shape[0])
        self.views.append({"x": int(x), "y": int(y), "size": size,
                           "origin": (x0, y0), "crop": crop, "route": route_on})
        res = {"_image": crop,
               "shows": {"x_min": box[0], "y_min": box[1], "x_max": box[2], "y_max": box[3]},
               "route_highlighted": route_on,
               "drawing_size": {"width": self.w, "height": self.h}}
        if route_on:
            r = self._routes.get((int(x) // 20, int(y) // 20))
            exits = _route.boundary_exits(r["mask"], box) if r is not None else []
            res["route_leaves_view_at"] = exits
            res["note"] = ("The route stays entirely inside this view - both ends "
                           "are visible here." if not exits else
                           "The route crosses the view edge at the points above; "
                           "view() there to see where it goes.")
        return res


TOOL_SPEC = """
find_label(text)                      -> where a line number or tag is. Fuzzy.
list_equipment()                      -> every equipment tag and its position.
nearest_equipment(x, y, k)            -> equipment closest to a point, with the
                                         direction to each. Use when trace_route
                                         finds nothing.
trace_route(x, y)                     -> extent of the piping run through (x,y),
                                         and any arrowheads found on it.
view(x, y, size, highlight_route)     -> LOOK at the drawing there. size 300-2560,
                                         default 1400. Returns an image.
answer(from, to, confidence, reasoning)
                                      -> finish. "from"/"to" must each be an
                                         equipment tag or "Off-page".
"""

SYSTEM = """You are tracing one pipe on a P&ID (Piping and Instrumentation Diagram).

Target line number: {target}
Drawing size: {w} x {h} pixels. All coordinates are ORIGINAL drawing pixels.
A single view can cover at most {max_view} x {max_view} pixels.

The target label has already been located at ({lx}, {ly}) and the FIRST VIEW below
is centred on it. {route_line}

Equipment visible in that view - endpoints must come from this list:
{candidates}

If the view already shows where both ends of the pipe go, call answer now.
Only look further if an end runs off the edge of the view. Tools:
{tools}
You have {max_views} view() calls left. Each view must be centred somewhere the
target pipe actually goes - use trace_route's extent, do not pick a point at random.

Rules:
- Inline valves and instruments are NOT endpoints; trace through them.
- If the pipe ends at an off-page connector arrow, the endpoint is "Off-page".
- "from" and "to" must be different. Both cannot be "Off-page".
- Do not guess: if you cannot determine an endpoint, answer "Unknown" for it.

Reply with ONE tool call as JSON and nothing else:
{{"tool": "<name>", "args": {{...}}}}
"""


def run_agent(image_path: str, target: str, out_dir: str = "outputs/agent",
              max_steps: int = 10, backend: str = "local",
              verbose: bool = True, max_view: int = 2560,
              at: Tuple[float, float] = None) -> Dict[str, Any]:
    """Let the model drive the pipeline. Returns the same shape as tier0.run()."""
    from src.vlm import local_qwen

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"cannot read image: {image_path}")

    box = Toolbox(image, target, max_view=max_view)
    h, w = image.shape[:2]

    # Seed with the fixed pipeline's opening move. Started blank, the model picks
    # its own coordinates and lands on uninformative parts of the drawing; the
    # label-centred crop is a prior worth keeping, and it can still move after.
    if at is not None:
        # Hand-given label position; see tier0.run(). Keeps a drawing whose text
        # OCR cannot read from collapsing at step one.
        bw = max(60.0, 7.0 * len(target))
        lab = {"text": target, "score": 1.0, "cx": float(at[0]), "cy": float(at[1]),
               "x": float(at[0]) - bw / 2, "y": float(at[1]) - 7.0, "w": bw, "h": 14.0}
    else:
        hits = locate_label(box.merged, target, 1)
        if not hits:
            return {"pipe": target, "from": "Unknown", "to": "Unknown",
                    "confidence": 0.0, "error": "label not found", "source": "agent_loop"}
        lab = hits[0]
    seed = box.view(x=lab["cx"], y=lab["cy"], size=min(1400, max_view))
    seed_img = seed.pop("_image")
    shows = seed["shows"]
    cands = equipment_candidates(box.merged, shows["x_min"], shows["y_min"],
                                 shows["x_max"], shows["y_max"])

    rt = box.trace_route(lab["cx"], lab["cy"])
    route_line = ("Its route is drawn ORANGE and any arrowhead on it is ringed RED; "
                  f"the route spans {rt['extent']}." if rt.get("found") else
                  "Its route could not be isolated, so nothing is highlighted.")

    messages = [{"role": "user", "content": [
        {"type": "text", "text": SYSTEM.format(
            target=target, w=w, h=h, tools=TOOL_SPEC, max_views=MAX_VIEWS - 1,
            max_view=max_view,
            lx=round(lab["cx"]), ly=round(lab["cy"]), route_line=route_line,
            candidates="\n".join(f"  - {c}" for c in cands))},
        {"type": "image", "image": seed_img},
        {"type": "text", "text": f"First view: {json.dumps(seed)}"}]}]

    trace, answer = [], None
    for step in range(max_steps):
        reply = local_qwen.chat(messages) if backend == "local" else {"error": "local only"}
        if "error" in reply:
            trace.append({"step": step, "error": reply["error"]})
            break

        name = str(reply.get("tool", "")).strip()
        args = reply.get("args") or {}
        trace.append({"step": step, "tool": name, "args": args})
        if verbose:
            print(f"  [{step}] {name}({json.dumps(args, ensure_ascii=False)[:90]})")

        if name == "answer":
            answer = args
            break

        messages.append({"role": "assistant",
                         "content": [{"type": "text", "text": json.dumps(reply)}]})

        fn = getattr(box, name, None)
        if fn is None or name.startswith("_"):
            result, img = {"error": f"no such tool: {name}",
                           "available": [t.split("(")[0].strip()
                                         for t in TOOL_SPEC.strip().splitlines()]}, None
        elif name == "view" and len(box.views) >= MAX_VIEWS:
            result, img = {"error": f"view budget spent ({MAX_VIEWS}); call answer now"}, None
        else:
            try:
                result = fn(**args)
            except TypeError as e:
                result = {"error": f"bad arguments: {e}"}
            except Exception as e:
                result = {"error": f"{type(e).__name__}: {e}"}
            img = result.pop("_image", None) if isinstance(result, dict) else None

        content = [{"type": "text",
                    "text": f"Result of {name}: {json.dumps(result, ensure_ascii=False)}"}]
        if img is not None:
            content.append({"type": "image", "image": img})
            content.append({"type": "text", "text":
                            "Above is that view. Continue, or call answer."})
        messages.append({"role": "user", "content": content})

    result = {"pipe": target,
              "from": (answer or {}).get("from", "Unknown"),
              "to": (answer or {}).get("to", "Unknown"),
              "confidence": (answer or {}).get("confidence", 0.0),
              "reasoning": (answer or {}).get("reasoning"),
              "source": "agent_loop",
              "steps": len(trace), "views": len(box.views), "trace": trace}

    for i, v in enumerate(box.views):
        cv2.imwrite(str(out / f"view{i}_{v['x']}_{v['y']}_{v['size']}.png"), v["crop"])
    (out / "trace.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _self_check():
    """Tools must be callable by name with JSON args and survive bad input."""
    img = np.full((600, 600, 3), 255, np.uint8)
    cv2.line(img, (50, 300), (550, 300), (0, 0, 0), 7)
    cv2.putText(img, "P-101", (200, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    box = Toolbox(img, "P-101")

    r = box.view(**{"x": 300, "y": 300, "size": 400})
    assert "_image" in r and r["_image"].shape[0] == 400, r.get("shows")
    assert r["shows"]["x_min"] == 100, r["shows"]

    assert box.view(x=300, y=300, size=50)["_image"].shape[0] == 200      # clamped up
    assert box.view(x=300, y=300, size=9999)["_image"].shape[0] == 600    # clamped to image
    assert Toolbox(img, "P-101", max_view=400).view(
        x=300, y=300, size=9999)["_image"].shape[0] == 400                # capped

    assert box.trace_route(x=300, y=300)["found"] is True
    assert box.trace_route(x=20, y=20)["found"] is False                  # blank corner

    assert "matches" in box.find_label(text="P-101")

    # a failed trace must hand back an action, not a dead end
    dead = box.trace_route(x=20, y=20)
    assert "next" in dead and "nearest_equipment" in dead["next"], dead

    ne = box.nearest_equipment(x=300, y=300)
    assert "nearest" in ne and isinstance(ne["nearest"], list)
    for e in ne["nearest"]:
        assert e["direction"] in ("up", "down", "left", "right"), e

    # a view holding the whole run reports no exits; a tight one reports some
    assert box.view(x=300, y=300, size=600).get("route_leaves_view_at") == []
    tight = box.view(x=300, y=300, size=300)
    assert {e["side"] for e in tight.get("route_leaves_view_at", [])} == {"left", "right"}, tight
    assert getattr(box, "nope", None) is None                            # unknown tool path
    print("agent self-check OK")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="VLM-driven agent loop for P&ID FROM-TO")
    p.add_argument("image", nargs="?"); p.add_argument("target", nargs="?")
    p.add_argument("--out", default="outputs/agent")
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--max-view", type=int, default=2560)
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    if a.self_check:
        _self_check()
    else:
        print(json.dumps({k: v for k, v in
                          run_agent(a.image, a.target, a.out, a.max_steps,
                                    max_view=a.max_view).items()
                          if k != "trace"}, indent=2, ensure_ascii=False))
