#!/usr/bin/env python3
"""Build the (gitignored) pilot manifest from Memento10k metadata.

Memento10k is non-commercial + no-redistribution: this writes to `configs/memento_manifest.local.csv`
(gitignored) and prints only counts — never the scores. Run it AFTER you've downloaded the dataset
via your access link (memento.csail.mit.edu). The metadata format varies by release, so this is
forgiving about key names; if it can't find the fields, it prints the keys it saw so you can map them.

    python3 scripts/build_memento_manifest.py --json path/to/memento_data.json \
        --video-dir /content/memento/videos --n 800 --split train
"""
import argparse, csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FNAME_KEYS = ("filename", "video", "video_name", "name", "url", "file")
SCORE_KEYS = ("mem_score", "memorability", "score", "mem", "short_term_memorability")


def first(d: dict, keys):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def records(obj):
    """Yield dict records from either a list or an id->dict mapping."""
    if isinstance(obj, list):
        yield from (r for r in obj if isinstance(r, dict))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                v.setdefault("filename", k)
                yield v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Memento metadata json (from your download)")
    ap.add_argument("--video-dir", default="/content/memento/videos",
                    help="dir where the clips live (in Colab), prefixed onto each filename")
    ap.add_argument("--n", type=int, default=0, help="cap to first N clips (0 = all)")
    ap.add_argument("--split", default="", help="optional split label written to every row")
    ap.add_argument("--out", default=str(ROOT / "configs" / "memento_manifest.local.csv"))
    args = ap.parse_args()

    obj = json.loads(Path(args.json).read_text())
    recs = list(records(obj))
    if not recs:
        sys.exit("no records found in the json (expected a list of dicts or an id->dict map)")

    rows, skipped = [], 0
    for i, r in enumerate(recs):
        fn, sc = first(r, FNAME_KEYS), first(r, SCORE_KEYS)
        if fn is None or sc is None:
            skipped += 1
            continue
        sid = Path(str(fn)).stem
        vpath = f"{args.video_dir.rstrip('/')}/{Path(str(fn)).name}"
        rows.append((sid, vpath, float(sc), args.split))
    if not rows:
        sys.exit(f"found {len(recs)} records but no filename+score; keys seen: {sorted(recs[0])}")
    if args.n:
        rows = rows[: args.n]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stimulus_id", "video_path", "score", "split"])
        w.writerows(rows)
    print(f"wrote {len(rows)} clips -> {args.out}  (skipped {skipped} without filename/score)")
    print("This file is gitignored (Memento is non-commercial + no-redistribution). Upload it to Colab.")


if __name__ == "__main__":
    main()
