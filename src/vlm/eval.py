"""Score the Tier 0 baseline against data/ground_truth.json.

This is the number to beat before building anything more elaborate.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.vlm.tier0 import run
from src.vlm.agent import run_agent


def _match(answer, accepted) -> bool:
    a = str(answer or "").strip().upper().replace(" ", "")
    return any(a == str(x).strip().upper().replace(" ", "") for x in accepted)


def main(gt_path="data/ground_truth.json", out_dir="outputs/eval", crop_size=1400,
         call_vlm=True, backend=None, agent=False, max_view=2560):
    gt = json.loads(Path(gt_path).read_text())
    print(f"[eval] 이미지 {gt['image']}  ->  출력 {out_dir}/")
    rows, n_from, n_to, n_both = [], 0, 0, 0

    for c in gt["cases"]:
        sub = f"{out_dir}/{c['pipe'].replace('/', '_')}"
        if agent:
            r = run_agent(gt["image"], c["pipe"], sub, backend=backend or "local",
                          verbose=False, max_view=max_view, at=c.get("at"))
            r["match_score"] = 1.0        # the agent locates the label via its own tool
        else:
            r = run(gt["image"], c["pipe"], sub, crop_size, call_vlm, backend, c, c.get("at"))
        ok_f, ok_t = _match(r.get("from"), c["from"]), _match(r.get("to"), c["to"])
        n_from += ok_f; n_to += ok_t; n_both += ok_f and ok_t
        rows.append({"pipe": c["pipe"], "located": r.get("match_score", 0) >= 0.8,
                     "from": r.get("from"), "to": r.get("to"),
                     "steps": r.get("steps"), "views": r.get("views"),
                     "ok_from": ok_f, "ok_to": ok_t, "error": r.get("error")})

    n = len(rows)
    print(f"\n{'pipe':<44} {'FROM':<12} {'TO':<12} 판정")
    print("-" * 82)
    for r in rows:
        mark = "OK" if r["ok_from"] and r["ok_to"] else ("partial" if r["ok_from"] or r["ok_to"] else "FAIL")
        print(f"{r['pipe']:<44} {str(r['from'])[:11]:<12} {str(r['to'])[:11]:<12} {mark}"
              + (f"  ({r['error']})" if r.get("error") else ""))
    print("-" * 82)
    print(f"\n결과 폴더: {out_dir}/  (배관별 01~04 PNG + result.json, 요약 eval.json)")
    print(f"locate {sum(r['located'] for r in rows)}/{n}   "
          f"FROM {n_from}/{n} ({n_from/n:.0%})   TO {n_to}/{n} ({n_to/n:.0%})   "
          f"BOTH {n_both}/{n} ({n_both/n:.0%})")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir, "eval.json").write_text(json.dumps(
        {"n": n, "from": n_from, "to": n_to, "both": n_both, "rows": rows},
        indent=2, ensure_ascii=False))
    return n_both / n


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--gt", default="data/ground_truth.json")
    p.add_argument("--out", default="outputs/eval")
    p.add_argument("--crop-size", type=int, default=1400)
    p.add_argument("--no-vlm", action="store_true")
    p.add_argument("--backend", default=None, choices=["local", "gemini"])
    p.add_argument("--agent", action="store_true", help="VLM drives the tools itself")
    p.add_argument("--max-view", type=int, default=2560, help="cap on view size")
    a = p.parse_args()
    main(a.gt, a.out, a.crop_size, not a.no_vlm, a.backend, a.agent, a.max_view)
