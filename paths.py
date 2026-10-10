from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_STORAGE_ROOT = PROJECT_ROOT.parent / f"{PROJECT_ROOT.name}-storage"
LOCAL_SETTINGS_FILE = PROJECT_ROOT / "paths.local.json"


def _load_local_settings() -> dict[str, Any]:
    if not LOCAL_SETTINGS_FILE.exists():
        return {}
    try:
        return json.loads(LOCAL_SETTINGS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


LOCAL_SETTINGS = _load_local_settings()


def _as_path(value: str | os.PathLike[str] | None) -> Path | None:
    if not value:
        return None
    return Path(value).expanduser()


def _storage_fallback(relative_path: str) -> Path:
    if DEFAULT_STORAGE_ROOT.exists():
        return DEFAULT_STORAGE_ROOT / relative_path
    return PROJECT_ROOT / relative_path


def _resolve_setting(env_key: str, settings_key: str, relative_path: str) -> Path:
    env_value = _as_path(os.environ.get(env_key))
    if env_value is not None:
        return env_value

    local_value = _as_path(LOCAL_SETTINGS.get(settings_key))
    if local_value is not None:
        return local_value

    return _storage_fallback(relative_path)


STORAGE_ROOT = _resolve_setting("UNISIGN_STORAGE_ROOT", "storage_root", "")
DATA_ROOT = _resolve_setting("UNISIGN_DATA_ROOT", "data_root", "data")
OUTPUT_ROOT = _resolve_setting("UNISIGN_OUTPUT_ROOT", "output_root", "out")
PRETRAINED_ROOT = _resolve_setting("UNISIGN_PRETRAINED_ROOT", "pretrained_root", "pretrained_weight")
CACHE_ROOT = _resolve_setting("UNISIGN_CACHE_ROOT", "cache_root", ".cache")
MT5_ROOT = _resolve_setting("UNISIGN_MT5_ROOT", "mt5_root", "pretrained_weight/mt5-base")
CE_CSL_ROOT = _resolve_setting("CE_CSL_ROOT", "ce_csl_root", "data/CE_CSL/CE-CSL")


def resolve_output_dir(raw_output_dir: str) -> str:
    if not raw_output_dir:
        return raw_output_dir

    normalized = raw_output_dir.replace("\\", "/").strip("/")
    if normalized == "out":
        return str(OUTPUT_ROOT)
    if normalized.startswith("out/"):
        return str(OUTPUT_ROOT / Path(normalized).relative_to("out"))

    return resolve_storage_path(raw_output_dir)


def resolve_storage_path(raw_path: str) -> str:
    if not raw_path:
        return raw_path

    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return str(candidate)

    normalized = raw_path.replace("\\", "/").strip("/")
    mapping = (
        ("out", OUTPUT_ROOT),
        ("pretrained_weight", PRETRAINED_ROOT),
        ("data", DATA_ROOT),
        (".cache", CACHE_ROOT),
    )
    for prefix, target_root in mapping:
        if normalized == prefix:
            return str(target_root)
        if normalized.startswith(prefix + "/"):
            return str(target_root / Path(normalized).relative_to(prefix))

    return str(PROJECT_ROOT / candidate)


def describe_path_config() -> dict[str, str]:
    return {
        "project_root": str(PROJECT_ROOT),
        "storage_root": str(STORAGE_ROOT),
        "data_root": str(DATA_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "pretrained_root": str(PRETRAINED_ROOT),
        "cache_root": str(CACHE_ROOT),
        "mt5_root": str(MT5_ROOT),
        "ce_csl_root": str(CE_CSL_ROOT),
    }
