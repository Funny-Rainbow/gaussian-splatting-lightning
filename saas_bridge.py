#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _dispatch(command: str, payload: dict) -> dict:
    from saas_api import (
        run_extract_tsdf_mesh,
        run_keyframes,
        run_prune_position_sigma,
        run_render_tsdf_frames,
        run_train,
        run_transform,
    )

    if command == "keyframes":
        return run_keyframes(payload)
    if command == "train":
        return run_train(payload)
    if command == "transform":
        return run_transform(payload)
    if command == "prune-position-sigma":
        return run_prune_position_sigma(payload)
    if command == "render-tsdf-frames":
        return run_render_tsdf_frames(payload)
    if command == "extract-tsdf-mesh":
        return run_extract_tsdf_mesh(payload)
    raise RuntimeError(f"Unsupported command: {command}")


def _normalize_success_payload(payload: dict) -> dict:
    if isinstance(payload, dict):
        if "ok" not in payload:
            return {"ok": True, **payload}
        return payload
    return {"ok": True, "result": payload}


def _error_payload(exc: Exception) -> dict:
    return {
        "ok": False,
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "stack": traceback.format_exc(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "keyframes",
            "train",
            "transform",
            "prune-position-sigma",
            "render-tsdf-frames",
            "extract-tsdf-mesh",
        ],
    )
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--response-json", required=True)
    args = parser.parse_args()

    request_path = Path(args.request_json)
    response_path = Path(args.response_json)

    try:
        payload = json.loads(request_path.read_text(encoding="utf-8"))
        result = _dispatch(args.command, payload)
        write_json(response_path, _normalize_success_payload(result))
        return 0
    except Exception as exc:
        failure = _error_payload(exc)
        try:
            write_json(response_path, failure)
        finally:
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
