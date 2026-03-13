import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Optional
from urllib.parse import urlparse

import numpy as np

from redundant_image_detection_sharpness_SaaS import video_mode


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _require_bearer_token(handler: BaseHTTPRequestHandler, token: Optional[str]) -> bool:
    if not token:
        return True
    auth = handler.headers.get("Authorization", "")
    if auth == f"Bearer {token}":
        return True
    _json_response(handler, 401, {"ok": False, "error": "unauthorized"})
    return False


def _build_args(req: dict) -> SimpleNamespace:
    input_path = req.get("input")
    if not input_path or not isinstance(input_path, str):
        raise ValueError("missing 'input' (video file path)")
    if not os.path.isfile(input_path):
        raise ValueError(f"input not found: {input_path}")

    output_dir = req.get("output_dir") or req.get("output-dir") or None
    fps = int(req.get("fps", 10))
    dist = float(req.get("dist", 0.1))
    ratio = float(req.get("ratio", 0.3))
    max_size = int(req.get("max_size", req.get("max-size", 1024)))
    save_max_long_edge = int(
        req.get(
            "save_max_long_edge",
            req.get("save-max-long-edge", req.get("max_long_edge", req.get("max-long-edge", 0)))
        )
    )
    start_number = int(req.get("start_number", req.get("start-number", 1)))
    filename_format = str(req.get("filename_format", req.get("filename-format", "%05d.jpg")))

    min_sharpness = float(req.get("min_sharpness", req.get("min-sharpness", 50.0)))
    sharpness_max_size = int(req.get("sharpness_max_size", req.get("sharpness-max-size", 640)))
    sharpness_grid = req.get("sharpness_grid", req.get("sharpness-grid", (4, 4)))
    if isinstance(sharpness_grid, (list, tuple)) and len(sharpness_grid) == 2:
        sharpness_grid = (int(sharpness_grid[0]), int(sharpness_grid[1]))
    else:
        # Kept only for backward compatibility; sharpness calculation no longer uses grid.
        sharpness_grid = (4, 4)
    sharpness_tile_percentile = float(
        req.get("sharpness_tile_percentile", req.get("sharpness-tile-percentile", 50.0))
    )
    sharpness_pre_blur_sigma = float(
        req.get("sharpness_pre_blur_sigma", req.get("sharpness-pre-blur-sigma", 0.0))
    )
    sharpness_downscale = float(req.get("sharpness_downscale", req.get("sharpness-downscale", 0.5)))
    if not (0.0 < sharpness_downscale <= 1.0):
        raise ValueError("sharpness_downscale must be in (0, 1]")

    return SimpleNamespace(
        input=input_path,
        output_dir=output_dir,
        fps=fps,
        dist=dist,
        ratio=ratio,
        max_size=max_size,
        save_max_long_edge=save_max_long_edge,
        start_number=start_number,
        filename_format=filename_format,
        min_sharpness=min_sharpness,
        sharpness_max_size=sharpness_max_size,
        sharpness_grid=sharpness_grid,
        sharpness_tile_percentile=sharpness_tile_percentile,
        sharpness_pre_blur_sigma=sharpness_pre_blur_sigma,
        sharpness_downscale=sharpness_downscale,
    )


def _is_keyframe_factory(dist: float, ratio: float):
    def is_keyframe(xy0, xy1):
        normalized_dist = np.linalg.norm(xy0 - xy1, axis=-1)
        over_threshold_mask = normalized_dist > dist
        over_threshold_ratio = over_threshold_mask.sum() / (over_threshold_mask.shape[0] + 1)
        return xy0.shape[0] < 32 or over_threshold_ratio > ratio

    return is_keyframe


def make_handler(token: Optional[str]):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path != "/extract":
                _json_response(self, 404, {"ok": False, "error": "not_found"})
                return
            if not _require_bearer_token(self, token):
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                _json_response(self, 400, {"ok": False, "error": "invalid_content_length"})
                return
            if length <= 0:
                _json_response(self, 400, {"ok": False, "error": "empty_body"})
                return

            try:
                body = self.rfile.read(length)
                req = json.loads(body.decode("utf-8"))
                if not isinstance(req, dict):
                    raise ValueError("body must be a JSON object")
                args = _build_args(req)
                is_keyframe = _is_keyframe_factory(args.dist, args.ratio)
                result = video_mode(args, is_keyframe, quiet=True)
                _json_response(self, 200, {"ok": True, **result})
            except ValueError as e:
                _json_response(self, 400, {"ok": False, "error": str(e)})
            except Exception as e:
                _json_response(self, 500, {"ok": False, "error": str(e)})

        def log_message(self, format, *args):
            return

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional bearer token. If set, require header: Authorization: Bearer <token>.",
    )
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.token))
    print(f"redundant API listening on http://{args.host}:{args.port}/extract")
    server.serve_forever()


if __name__ == "__main__":
    main()
