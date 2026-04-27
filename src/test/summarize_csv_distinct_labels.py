"""
Summarize distinct labels per CSV field.

Input: a CSV exported by `run_video_analysis_v2.py` (utf-8-sig).
Output: JSON mapping { field: {distinct_count, label_list} }.

Heuristics:
- Cells joined by " | " are treated as multi-value lists (same as _flat() in exporter).
- `obs_frames` is treated as multi-value list split by comma.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Set


def _split_values(field: str, raw: str) -> List[str]:
    s = (raw or "").strip()
    if not s:
        return []

    if field == "obs_frames":
        parts = [p.strip() for p in s.split(",")]
        return [p for p in parts if p]

    # Exporter uses " | " to flatten list fields.
    if " | " in s:
        parts = [p.strip() for p in s.split(" | ")]
        return [p for p in parts if p]

    return [s]


def summarize_csv(csv_path: Path) -> Dict[str, Dict[str, object]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row")

        # Track labels per column.
        acc: Dict[str, Set[str]] = {k: set() for k in reader.fieldnames}

        for row in reader:
            for k in acc.keys():
                raw = row.get(k, "")
                for v in _split_values(k, raw):
                    acc[k].add(v)

    out: Dict[str, Dict[str, object]] = {}
    for k, labels in acc.items():
        lst = sorted(labels)
        out[k] = {"distinct_count": len(lst), "label_list": lst}
    return out


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        dest="csv_path",
        required=True,
        help="Path to run_video_analysis_v2.csv",
    )
    ap.add_argument(
        "--out",
        dest="out_path",
        default="",
        help="Output json path (default: alongside csv, suffix _labels.json)",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    csv_path = Path(args.csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(str(csv_path))

    out_path = Path(args.out_path).expanduser().resolve() if args.out_path else csv_path.with_suffix("")
    if not args.out_path:
        out_path = Path(str(out_path) + "_labels.json")

    summary = summarize_csv(csv_path)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK. Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

