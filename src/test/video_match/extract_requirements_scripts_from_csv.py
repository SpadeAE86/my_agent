"""
extract_requirements_scripts_from_csv.py

Input CSV is a "matrix" layout:
- Row1: 序号,1,2,...,100
- Row2+: first column is a field name (主题/标题/中段混剪/车型/...)
         columns 2.. are values for script 1..100.

This script extracts four reusable fields into a JSON file:
- 主题
- 标题
- 中段混剪
- 车型

Output: 需求脚本.json (UTF-8, ensure_ascii=False)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


CSV_PATH = Path(r"C:\Users\25065\Downloads\0421数字人（100条）  - 脚本.csv")
OUT_PATH = Path(__file__).resolve().parent / "需求脚本.json"


FIELD_MAP = {
    "主题": "topic",
    "标题": "title",
    "中段混剪": "mid_mix",
    "车型": "car_model",
}


def _strip(v: Any) -> str:
    return str(v).strip() if v is not None else ""


def _load_matrix_rows(path: Path) -> Dict[str, List[str]]:
    """
    Returns: {row_key: [v1..vN]} where N matches the numbered columns.
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    rows: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        for line in r:
            if not line:
                continue
            key = _strip(line[0])
            if not key:
                continue
            vals = [_strip(x) for x in (line[1:] if len(line) > 1 else [])]
            rows[key] = vals
    return rows


def _infer_count(header_row: Optional[List[str]], rows: Dict[str, List[str]]) -> int:
    if header_row:
        # header_row like: ["序号","1","2",...]
        nums = [x for x in header_row[1:] if _strip(x)]
        if nums and all(n.isdigit() for n in nums):
            return len(nums)
    # fallback: longest row length
    return max((len(v) for v in rows.values()), default=0)


def main():
    # Read file twice: first to keep header row, then parse into dict.
    header: List[str] = []
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        header = next(r, []) or []

    rows = _load_matrix_rows(CSV_PATH)
    n = _infer_count(header, rows)
    if n <= 0:
        raise RuntimeError("Could not infer script count from CSV.")

    # Extract wanted rows
    extracted: Dict[str, List[str]] = {}
    for cn_key in FIELD_MAP.keys():
        extracted[cn_key] = rows.get(cn_key, [])

    items: List[Dict[str, Any]] = []
    for i in range(n):
        item: Dict[str, Any] = {"id": i + 1}
        for cn_key, out_key in FIELD_MAP.items():
            vals = extracted.get(cn_key) or []
            item[out_key] = vals[i] if i < len(vals) else ""
        # drop completely empty rows (rare but safe)
        if any(_strip(item.get(k)) for k in ("topic", "title", "mid_mix", "car_model")):
            items.append(item)

    payload = {
        "source_csv": str(CSV_PATH),
        "count": len(items),
        "items": items,
    }

    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote:", str(OUT_PATH))
    print("Count:", len(items))


if __name__ == "__main__":
    main()

