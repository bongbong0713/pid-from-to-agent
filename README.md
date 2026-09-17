# P&ID FROM-TO Agent

Given a P&ID drawing and a pipe line number, identify which equipment the pipe runs
**from** and **to**.

> **Status: research prototype, and the numbers do not hold up across drawings.**
> On the first test drawing it identifies both endpoints 86% of the time; on a
> second, harder drawing that falls to 22%. Flow direction is never better than
> chance. Read [Measured results](#measured-results) before relying on any output.

---

## Structure

Two paths share the same building blocks.

**`tier0`** — a linear pipeline with a single VLM call. No tool use, no iteration;
the model never chooses where to look. This is the stronger of the two.

```
image + target line number
  │
  ├─ 1. ocr_words()            tesseract --psm 11 on a 2x upscale
  │                            → words with bboxes, in ORIGINAL image coordinates
  │
  ├─ 2. merge_fragments()      glue words sharing a baseline
  │                            "10-LCV-110" + "10-20000-NPH-250" + "-W10CB01-HC-50"
  │                            → one label
  │
  ├─ 3. locate_label()         difflib ranking against the target
  │                            tolerates OCR damage ("19-30001" read as "18-30001")
  │                            (or skipped entirely — see `at` below)
  │
  ├─ 4. pipe_route()           distance transform → the thick connected component
  │     arrowheads_on_route()  ink swells on that route only
  │     render()               recolour the route orange, ring each arrowhead
  │
  ├─ 5. build_crop()           square crop, coordinate grid, target boxed red
  │
  ├─ 6. equipment_candidates() tags in the crop, told apart from line numbers by
  │                            type size where the drawing sets them differently
  │
  └─ 7. ask_vlm()              Qwen2.5-VL-7B locally, or Gemini over REST
                               → {from, to, confidence, reasoning}
```

**`agent`** — the same steps exposed as tools the model invokes itself
(`find_label`, `list_equipment`, `nearest_equipment`, `trace_route`, `view`,
`answer`), seeded with tier0's opening crop and then free to move. It works
mechanically but scores lower everywhere it was measured; see
[The agent loop](#the-agent-loop).

Everything runs in one coordinate space — the original image. Nothing is resized
except inside OCR, which divides its coordinates back out.

### Files

| File | Lines | Role |
|---|---|---|
| [`src/vlm/tier0.py`](src/vlm/tier0.py) | ~470 | The pipeline above, plus CLI and stage visualizations |
| [`src/vlm/agent.py`](src/vlm/agent.py) | ~370 | Tool definitions and the VLM-driven loop |
| [`src/vlm/route.py`](src/vlm/route.py) | ~150 | Route extraction, arrowhead marking, boundary exits |
| [`src/vlm/local_qwen.py`](src/vlm/local_qwen.py) | ~150 | Local Qwen2.5-VL loader, single-turn and multi-turn |
| [`src/vlm/eval.py`](src/vlm/eval.py) | ~65 | Scores either path against an answer key |
| [`src/vlm/smoke.py`](src/vlm/smoke.py) | ~25 | Minimal end-to-end check |

`tier0.py`, `agent.py` and `route.py` each carry `--self-check` assertions, including
regression tests for the two bugs recorded below.

---

## Setup

Needs **OpenCV**, **requests**, and the **`tesseract`** binary. For the local model
also **torch**, **transformers ≥ 5.11**, **accelerate**, **bitsandbytes**.

> `requirements.txt` is left over from the previous implementation and **does not
> install** on Python 3.12 — `paddlepaddle==2.6.2` fails to build. Nothing in
> `src/vlm/` uses it.

```bash
python3 -c "from huggingface_hub import snapshot_download; \
            snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct')"   # ~16GB

export GOOGLE_API_KEY=...          # or use hosted Gemini instead
```

### Run

```bash
# Score a drawing.  bf16 is worth one case over 4-bit; see below.
LOCAL_VLM_4BIT=0 python3 src/vlm/eval.py --backend local --out outputs/run1
LOCAL_VLM_4BIT=0 python3 src/vlm/eval.py --gt data/ground_truth2.json \
                                         --crop-size 1100 --out outputs/run2
LOCAL_VLM_4BIT=0 python3 src/vlm/eval.py --agent --backend local --out outputs/run3

# One pipe, and crops/visualizations without calling a model
python3 src/vlm/tier0.py data/sample_pid.png "16-20001-NPH-250 -W10CB01-HC-50"
python3 src/vlm/tier0.py data/sample_pid.png "<line number>" --no-vlm

python3 src/vlm/tier0.py --self-check && python3 src/vlm/agent.py --self-check \
  && python3 src/vlm/route.py
```

| Variable | Default | Meaning |
|---|---|---|
| `VLM_BACKEND` | `local` | `local` or `gemini` |
| `LOCAL_VLM_4BIT` | `1` | `0` for bf16 |
| `LOCAL_VLM_DEVICE` | `cuda:0` (4-bit) / `auto` (bf16) | device map |
| `LOCAL_VLM_MAX_PIXELS` | 2560² | crop resolution ceiling |
| `GOOGLE_API_KEY`, `MODEL_NAME` | — | Gemini backend |

### Answer keys

`data/ground_truth.json` and `data/ground_truth2.json` list the pipes and their
accepted answers. A case may also carry `"at": [x, y]`, a hand-given label
position; the pipeline then skips OCR and only the crop-and-reason half is
measured. The second drawing needs this — its line numbers are 6-8px tall and
tesseract reads 2 of 14.

### Output

Each pipe writes to `outputs/<run>/<pipe>/`: `01_ocr_candidates.png`,
`02_crop_region.png`, `03_vlm_input.png` (the exact bytes the model sees),
`04_result.png` (prediction **and** ground truth, colour-coded PASS/PARTIAL/FAIL)
and `result.json`. Published sets are under [`outputs/`](outputs/).

---

## Measured results

Qwen2.5-VL-7B bf16, one 16GB GPU, ~6s per pipe. "endpoint pair" ignores which end
is which; "with direction" is the number that counts.

| Drawing | Pipes | endpoint pair | with direction |
|---|---|---|---|
| 1 — [`sample_pid.png`](data/sample_pid.png) 2201×1701, crop 1400 | 7 | **6/7 (86%)** | **3/7 (43%)** |
| 2 — [`sample_pid2.jpg`](data/sample_pid2.jpg) 2201×1614, crop 1100, positions given | 9 of 10 | **2/9 (22%)** | **1/9 (11%)** |

Drawing 1 is public domain (Wikimedia Commons, *RI Sample ISO.png*). Drawing 2 is
CC BY-SA 3.0 (Wikimedia Commons, *P&ID.JPG*, Bewdley Technical Services Ltd /
Alistair Chemicals, dwg 190A.1.002). One of drawing 2's ten cases was lost when the
run was killed before writing its summary.

**The gap between the two drawings is the main result.** 86% was read off a single
drawing and does not generalise: drawing 2 sets its text at 6-8px against drawing
1's 13px, packs in roughly sixty instruments, and yields five equipment tags to OCR
where it actually has seven. On drawing 2 the model repeatedly answered with tags
that were in the candidate list but not connected to the target pipe — `MP02S` three
times, `MK01S` three times.

On drawing 1, three of the four failures were **pure FROM/TO swaps**: the right two
endpoints in the wrong order. 43% on a binary choice is indistinguishable from
guessing. Treat ordering as unverified.

---

## What did not work

### Flow direction — four attempts, all stuck at 43%

**1 — Zoom on arrowheads.** Arrowheads are ~30px and legible at 3-4× zoom, but asked
to read one the model scored 2-3/5. Circling the specific arrow lifted this to
**4/5**, so the failure is picking *which* triangle is a flow arrow, not seeing it —
a P&ID is full of triangles (check valves, pump impellers, motor glyphs). Automating
the marking is what fails; see attempt 4.

**2 — P&ID conventions in the prompt.** Rules for pump suction/discharge, suction
being a size larger, `STM` supplying and condensate returning. Result **1/7, far
worse**. The model stopped looking and fabricated observations to fit the rules —
claiming a service code the line did not have, calling a top inlet a bottom outlet.
Reverted.

**3 — Constraining the answer set** to tags visible in the crop. No effect on
drawing 1 (2/7 → 2/7). On drawing 2 it actively hurt: the model picked unconnected
tags off the list.

**4 — Route extraction with automatic arrow marking.** Kept, because it lifted
endpoint identification on drawing 1 from 5/7 to 6/7. It does not help direction:
the routes carry no readable arrow — some segments have no flow arrow at all, and
off-page connectors are hollow outlines with no ink swell to detect.

One thing that did help: **bf16 over 4-bit NF4**, worth one case.

### The agent loop

Exposing the pipeline as tools works mechanically — the model calls `find_label`,
`trace_route`, `view`, moves, looks again, and answers, and inspection of its traces
shows **every one of its 17 view calls landed on the target pipe's route**. It still
scores below the fixed pipeline at every setting measured:

| Drawing 1 | endpoint pair | with direction |
|---|---|---|
| tier0, crop 1400 | 6/7 (86%) | 3/7 (43%) |
| agent | 3/7 (43%) | 1/7 (14%) |

Three revisions each fixed a real defect — seeding with tier0's opening crop took it
from 1/7 to 4/7 on pair; `nearest_equipment` removed the re-viewing-the-same-spot
stalls and the `Unknown` answers — without moving the final score.

The standing argument for it was that a larger drawing would eventually need
navigation. **A crop-ratio sweep on drawing 1 does not support that**: shrinking the
crop to simulate a bigger sheet, the agent never overtakes, and at 26% coverage both
collapse together.

| crop | coverage | tier0 pair | agent pair |
|---|---|---|---|
| 1400 | 64% | 6/7 (86%) | 2/7 (29%) |
| 1120 | 51% | 6/7 (86%) | 2/7 (29%) |
| 570 | 26% | 0/7 (0%) | 0/7 (0%) |

That sweep shrinks the view rather than enlarging the sheet, so a real 10000px
drawing at the same coverage would still show far more per view. The question is not
settled, but the evidence currently runs against the agent.

### Bugs a second drawing exposed

Both were latent on drawing 1 and are fixed with regression tests:

- `locate_label` promoted any substring match to 0.9. Every single character of the
  target is trivially "contained" in it, so one-character OCR scraps (`T`, `a`, `0`)
  outranked every real candidate once OCR quality dropped.
- `equipment_candidates` returned an empty list on a drawing whose tags are written
  `SC 02` (space, no dash) in a uniform type size — while the prompt presented that
  list as the only allowed answers, which would have forced every answer to
  `Off-page`.

### Why the previous implementation was replaced

The earlier PaddleOCR → Hough → NetworkX pipeline could not return a correct answer,
confirmed by running it: Hough produced **93 segments for 3 pipes with 0% of their
endpoints coincident**, so the graph stayed in 186 disconnected pieces; pipe-label
nodes were wired to freshly minted line nodes connected to nothing; the tracer tested
`type == "equipment"` while the builder stored `"Reactor"`/`"Pump"`; every edge was
inserted both ways, making `from` always equal `to`; and Hough ran on a resized image
while OCR ran on the original, so the pixel thresholds compared two coordinate
spaces. `route.py` replaces Hough with a distance transform on stroke thickness.

---

## Limitations

- **Direction is unreliable** — 43% and 11% on the two drawings.
- **Endpoint accuracy is drawing-dependent**, 86% to 22%. Two drawings is not a
  sample; assume any number here is provisional.
- **OCR fails below ~10px text.** Drawing 2 needs hand-given label positions.
- **Route extraction covers 4 of 7 pipes on drawing 1.** Thin service lines are drawn
  at instrument weight; lowering the threshold fuses the sheet into one component.
- **Hollow off-page connector arrows are never detected** — no ink swell.
- The answer keys are one person's reading. Utility directions on drawing 2 (steam
  supply, cooling water supply/return) were taken from service codes, as that drawing
  shows no arrows on those runs.
- **Drawings without line numbers are out of scope.** The entry point is a line
  number; a drawing tagged only with equipment and valve numbers has nothing to look
  up.

## If you pick this up next

1. **Find out whether resolution is the cause** of the drawing-2 collapse: upscale it
   2-3× and re-measure. If that recovers the score, text size is the bottleneck; if
   not, it is drawing complexity.
2. **Try a larger model.** Model size is untested against the direction bottleneck —
   specifically whether it can pick the flow arrow out of the other triangles.
3. **Make the output honest about what is known.** Report the endpoint pair as a
   first-class result and mark direction low-confidence, instead of an ordered pair
   that is right 43% of the time on a good drawing and 11% on a harder one.
