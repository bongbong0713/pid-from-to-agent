"""Extract the physical route of one pipe, and highlight it for the VLM.

Two things were measured to be the bottleneck, and both are localisation, not
perception:
  - which of many black lines is the target pipe
  - which of many triangular symbols is its flow arrowhead
Circling the arrow lifted direction reading from 2/5 to 4/5, so the job here is
to produce those marks.

Routes come from stroke thickness, not Hough: process piping is drawn heavier
than instrument lines, and a distance transform separates them cleanly. Hough
fragmented the same drawing into 93 segments that shared 0% of their endpoints.
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def ink_distance(image: np.ndarray) -> np.ndarray:
    """Per-pixel half-stroke-width of the drawing's ink."""
    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.distanceTransform((g < 128).astype(np.uint8), cv2.DIST_L2, 5)


def _components(dist: np.ndarray, th: float) -> Tuple[int, np.ndarray, np.ndarray]:
    core = (dist >= th).astype(np.uint8)
    # Re-inflate the eroded core so a run survives the valves that interrupt it.
    thick = cv2.dilate(core, np.ones((int(th * 2) + 3,) * 2, np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(thick, 8)
    return n, lab, stats


def pipe_route(image: np.ndarray, lx: float, ly: float,
               min_area: int = 1500, max_ink_frac: float = 0.03) -> Optional[Dict[str, Any]]:
    """Mask of the piping run that the label at (lx, ly) sits on.

    Thresholds are tried heavy-first: the heavy one isolates main process lines,
    but small service lines (DN150 here) are drawn thin and only appear lower
    down, so fall back rather than return nothing.
    """
    dist = ink_distance(image)
    for th in (2.5, 2.0, 1.6):
        n, lab, stats = _components(dist, th)
        win = lab[max(0, int(ly) - 55):int(ly) + 55, max(0, int(lx) - 70):int(lx) + 70]
        ids, counts = np.unique(win[win > 0], return_counts=True)
        if len(ids) == 0:
            continue
        cid = int(ids[np.argmax(counts)])
        area = int(stats[cid, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        # Below ~2.0 the whole drawing fuses into one blob that "touches"
        # everything; such a component localises nothing, so decline it.
        if area > max_ink_frac * dist.size:
            continue
        return {"mask": (lab == cid), "area": area, "threshold": th,
                "bbox": tuple(int(stats[cid, k]) for k in
                              (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                               cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))}
    return None


def arrowheads_on_route(image: np.ndarray, mask: np.ndarray,
                        min_area: int = 8,
                        exclude: Tuple[float, float, float, float] = None) -> List[Tuple[int, int]]:
    """Points on the route where the ink swells — filled arrowheads.

    Restricting to the route is the whole point: over the full drawing the same
    test also returns check valves, pump impellers and the motor glyph.
    Hollow off-page connectors carry no swell and are not found here.
    """
    dist = ink_distance(image)
    swell = ((dist >= 3.2) & mask).astype(np.uint8)
    if exclude is not None:
        # Label text is as heavy as an arrowhead and sits right on the route.
        ex, ey, ew, eh = (int(v) for v in exclude)
        pad = 12
        swell[max(0, ey - pad):ey + eh + pad, max(0, ex - pad):ex + ew + pad] = 0
    n, lab, stats, cent = cv2.connectedComponentsWithStats(swell, 8)
    return [tuple(cent[i].astype(int)) for i in range(1, n)
            if stats[i, cv2.CC_STAT_AREA] >= min_area]


def render(image: np.ndarray, mask: np.ndarray, arrows: List[Tuple[int, int]],
           label_box: Tuple[float, float, float, float] = None,
           tint: Tuple[int, int, int] = (0, 200, 255)) -> np.ndarray:
    """Tint the route and ring each arrowhead, so the model is told what to read."""
    out = image.copy()
    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    out[mask & (g < 128)] = tint          # the route's own ink, recoloured
    faint = mask & (g >= 128)             # a soft halo so the run reads as one path
    out[faint] = (0.22 * np.array(tint) + 0.78 * out[faint]).astype(np.uint8)
    for (ax, ay) in arrows:
        cv2.circle(out, (int(ax), int(ay)), 30, (0, 0, 255), 3)
    if label_box is not None:
        x, y, w, h = label_box
        cv2.rectangle(out, (int(x) - 6, int(y) - 6), (int(x + w) + 6, int(y + h) + 6),
                      (255, 0, 255), 3)
    return out


def touched_equipment(mask: np.ndarray, anchors: Dict[str, Tuple[float, float]],
                      radius: int = 120) -> List[str]:
    """Equipment whose tag sits next to the route.

    A vessel outline is drawn as heavily as its piping, so it joins the same
    component - which is exactly the connectivity the old Hough graph never got.
    """
    h, w = mask.shape
    hits = []
    for tag, (ax, ay) in anchors.items():
        y0, y1 = max(0, int(ay) - radius), min(h, int(ay) + radius)
        x0, x1 = max(0, int(ax) - radius), min(w, int(ax) + radius)
        if mask[y0:y1, x0:x1].any():
            hits.append(tag)
    return sorted(hits)


def boundary_exits(mask: np.ndarray, box: Tuple[int, int, int, int],
                   min_run: int = 4) -> List[Dict[str, Any]]:
    """Where the run crosses the edges of `box` (x0, y0, x1, y1).

    This is the clue the model needs when a view does not contain both ends:
    it says which way the pipe keeps going, so the next view has somewhere to be.
    """
    x0, y0, x1, y1 = box
    h, w = mask.shape
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    out = []
    edges = {"left": (mask[y0:y1, x0], y0, "y"), "right": (mask[y0:y1, x1 - 1], y0, "y"),
             "top": (mask[y0, x0:x1], x0, "x"), "bottom": (mask[y1 - 1, x0:x1], x0, "x")}
    for side, (strip, offset, axis) in edges.items():
        idx = np.nonzero(strip)[0]
        if len(idx) == 0:
            continue
        for grp in np.split(idx, np.nonzero(np.diff(idx) > 3)[0] + 1):
            if len(grp) < min_run:
                continue
            c = int(grp.mean()) + offset
            out.append({"side": side, "x": c if axis == "x" else (x0 if side == "left" else x1),
                        "y": c if axis == "y" else (y0 if side == "top" else y1)})
    return out


def _self_check():
    """Route extraction must find a run for the target and keep arrow hunting on it."""
    img = np.full((400, 400, 3), 255, np.uint8)
    cv2.line(img, (50, 200), (350, 200), (0, 0, 0), 7)          # the pipe
    cv2.line(img, (50, 60), (350, 60), (0, 0, 0), 1)            # a thin instrument line
    cv2.drawContours(img, [np.array([[200, 185], [200, 215], [235, 200]])],
                     -1, (0, 0, 0), -1)                          # arrowhead on the pipe
    cv2.drawContours(img, [np.array([[200, 50], [200, 70], [225, 60]])],
                     -1, (0, 0, 0), -1)                          # decoy off the pipe

    r = pipe_route(img, 200, 200, min_area=500)
    assert r is not None, "route not found"
    assert r["mask"][200, 100], "route mask misses the pipe"
    assert not r["mask"][60, 100], "thin instrument line leaked into the route"

    arrows = arrowheads_on_route(img, r["mask"])
    assert arrows, "arrowhead on the route not found"
    assert all(abs(ay - 200) < 40 for _, ay in arrows), f"decoy picked up: {arrows}"

    assert touched_equipment(r["mask"], {"X-1": (60, 200), "Y-2": (60, 60)}, 40) == ["X-1"]

    # a line too thin to isolate is reported as absent, never guessed at
    assert pipe_route(img, 100, 60, min_area=500) is None, "thin line should not pass"

    ex = boundary_exits(r["mask"], (100, 150, 300, 260))
    assert {e["side"] for e in ex} == {"left", "right"}, ex
    assert all(abs(e["y"] - 200) < 15 for e in ex), ex
    print("route self-check OK")


if __name__ == "__main__":
    _self_check()
