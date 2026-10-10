from __future__ import annotations

import sys
from pathlib import Path


def add_vendor_path() -> Path:
    base_dir = Path(__file__).resolve().parent
    candidates = [base_dir / ".vendor_py", base_dir / ".vendor"]
    for vendor_dir in candidates:
        if vendor_dir.exists() and str(vendor_dir) not in sys.path:
            sys.path.insert(0, str(vendor_dir))
            return vendor_dir
    return candidates[0]
