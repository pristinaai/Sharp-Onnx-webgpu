#!/usr/bin/env python3
"""Local SHARP inference server (native PyTorch).

Use this when browser ONNX/WASM runs out of memory (~4 GB heap limit).
The web UI can call this server instead of running ORT in the browser.

  source .venv-export/bin/activate
  python scripts/inference_server.py --sharp-repo vendor/ml-sharp

Then in the web app, switch inference mode to "Local Python server".
"""

from __future__ import annotations

import argparse
import cgi
import io
import json
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

_predictor = None
_device = None
_torch = None
_F = None
_unproject_gaussians = None
_save_ply = None
_io = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sharp-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    return parser.parse_args()


def import_sharp(sharp_repo: Path):
    src_path = sharp_repo / "src"
    if not src_path.exists():
        raise FileNotFoundError(f"Missing {src_path}")
    sys.path.insert(0, str(src_path))

    import numpy as np
    import torch
    import torch.nn.functional as F

    from sharp.models import PredictorParams, create_predictor
    from sharp.utils import io as sharp_io
    from sharp.utils.gaussians import save_ply, unproject_gaussians

    return torch, F, create_predictor, PredictorParams, sharp_io, save_ply, unproject_gaussians


def predict_image(predictor, image, f_px: float, device, torch, F, unproject_gaussians):
    """Same logic as sharp.cli.predict.predict_image without gsplat/render imports."""
    with torch.no_grad():
        internal_shape = (1536, 1536)

        image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
        _, height, width = image_pt.shape
        disparity_factor = torch.tensor([f_px / width]).float().to(device)

        image_resized_pt = F.interpolate(
            image_pt[None],
            size=(internal_shape[1], internal_shape[0]),
            mode="bilinear",
            align_corners=True,
        )

        gaussians_ndc = predictor(image_resized_pt, disparity_factor)

        intrinsics = (
            torch.tensor(
                [
                    [f_px, 0, width / 2, 0],
                    [0, f_px, height / 2, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
            .float()
            .to(device)
        )
        intrinsics_resized = intrinsics.clone()
        intrinsics_resized[0] *= internal_shape[0] / width
        intrinsics_resized[1] *= internal_shape[1] / height

        return unproject_gaussians(
            gaussians_ndc,
            torch.eye(4).to(device),
            intrinsics_resized,
            internal_shape,
        )


def pick_device(torch, requested: str):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_predictor(torch, create_predictor, predictor_params_cls, checkpoint_path, device):
    predictor = create_predictor(predictor_params_cls())
    if checkpoint_path is None:
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
    else:
        try:
            state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
        except TypeError:
            state_dict = torch.load(checkpoint_path, map_location=device)
    predictor.load_state_dict(state_dict)
    predictor.eval()
    predictor.to(device)
    return predictor


def prune_gaussians(gaussians, opacity_threshold: float, max_gaussians: int, torch):
    opacities = gaussians.opacities[0]
    keep = opacities >= opacity_threshold
    if int(keep.sum()) == 0:
        keep = torch.ones_like(opacities, dtype=torch.bool)

    indices = torch.where(keep)[0]
    if max_gaussians > 0 and indices.numel() > max_gaussians:
        top = torch.topk(opacities[indices], k=max_gaussians).indices
        indices = torch.sort(indices[top]).values

    def sel(tensor):
        return tensor[:, indices, ...]

    return gaussians._replace(
        mean_vectors=sel(gaussians.mean_vectors),
        singular_values=sel(gaussians.singular_values),
        quaternions=sel(gaussians.quaternions),
        colors=sel(gaussians.colors),
        opacities=sel(gaussians.opacities),
    ), int(indices.numel()), int(opacities.numel())


def gaussians_to_ply_bytes(save_ply, gaussians, f_px: float, width: int, height: int) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".ply") as tmp:
        path = Path(tmp.name)
        save_ply(gaussians, f_px, (height, width), path)
        return path.read_bytes()


def run_predict(
    image_bytes: bytes,
    focal_px: float | None,
    opacity_threshold: float,
    max_gaussians: int,
) -> dict[str, Any]:
    global _predictor, _device, _torch, _F, _unproject_gaussians, _save_ply, _io

    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        tmp.write(image_bytes)
        tmp.flush()
        image, _, estimated_f_px = _io.load_rgb(Path(tmp.name))

    height, width = image.shape[:2]
    if focal_px is None or focal_px <= 0:
        focal_px = float(estimated_f_px)

    gaussians = predict_image(
        _predictor, image, float(focal_px), _device, _torch, _F, _unproject_gaussians
    )
    pruned, selected, total = prune_gaussians(
        gaussians, opacity_threshold, max_gaussians, _torch
    )
    ply_bytes = gaussians_to_ply_bytes(_save_ply, pruned, float(focal_px), width, height)

    return {
        "ply_bytes": ply_bytes,
        "selected_gaussians": selected,
        "total_gaussians": total,
        "width": width,
        "height": height,
        "focal_px": float(focal_px),
    }


class SharpHandler(BaseHTTPRequestHandler):
    server_version = "SharpLocalInference/0.1"

    def log_message(self, fmt: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def _send_cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_cors()
        self.end_headers()

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/health":
            payload = json.dumps(
                {"ok": True, "device": str(_device), "backend": "pytorch"}
            ).encode("utf-8")
            self.send_response(200)
            self._send_cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_error(404, "Not found")

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/predict":
            self.send_error(404, "Not found")
            return

        try:
            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in content_type:
                self.send_error(400, "Expected multipart/form-data")
                return

            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": content_type,
                },
            )

            image_field = form["image"] if "image" in form else None
            if image_field is None or not image_field.file:
                self.send_error(400, "Missing image field")
                return

            image_bytes = image_field.file.read()
            focal_px = float(form.getvalue("focal_px") or 0)
            opacity_threshold = float(form.getvalue("opacity_threshold") or 0.02)
            max_gaussians = int(form.getvalue("max_gaussians") or 200000)

            result = run_predict(image_bytes, focal_px or None, opacity_threshold, max_gaussians)
            ply_bytes = result["ply_bytes"]

            self.send_response(200)
            self._send_cors()
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("X-Selected-Gaussians", str(result["selected_gaussians"]))
            self.send_header("X-Total-Gaussians", str(result["total_gaussians"]))
            self.send_header("X-Focal-Px", str(result["focal_px"]))
            self.send_header("Content-Length", str(len(ply_bytes)))
            self.end_headers()
            self.wfile.write(ply_bytes)
        except Exception as exc:
            msg = str(exc).encode("utf-8")
            self.send_response(500)
            self._send_cors()
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)


def main() -> None:
    global _predictor, _device, _torch, _F, _unproject_gaussians, _save_ply, _io

    args = parse_args()
    sharp_repo = args.sharp_repo.expanduser().resolve()

    _torch, _F, create_predictor, predictor_params_cls, _io, _save_ply, _unproject_gaussians = import_sharp(
        sharp_repo
    )

    _device = pick_device(_torch, args.device)
    print(f"Loading SHARP on {_device}…")
    _predictor = load_predictor(
        _torch,
        create_predictor,
        predictor_params_cls,
        args.checkpoint.expanduser().resolve() if args.checkpoint else None,
        _device,
    )
    print(f"Listening on http://{args.host}:{args.port}")
    print("  GET  /health")
    print("  POST /predict  (multipart: image, focal_px, opacity_threshold, max_gaussians)")

    server = ThreadingHTTPServer((args.host, args.port), SharpHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
