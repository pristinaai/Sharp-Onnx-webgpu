#!/usr/bin/env python3
"""Export Apple SHARP to a browser-friendly ONNX predictor graph.

This exporter emits raw SHARP predictor outputs (Gaussians in NDC-aligned space).
The web worker performs NDC->metric conversion in JavaScript, avoiding ONNX SVD
which is poorly supported in browser runtimes.

Inputs:
  - image: float32 [1, 3, 1536, 1536] in [0, 1]
  - disparity_factor: float32 [1] (f_px / image_width)

Outputs:
  - mean_vectors_ndc, singular_values_ndc, quaternions_ndc, colors, opacities

License: SHARP model weights are research-only — see apple/ml-sharp LICENSE_MODEL.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--sharp-repo",
        type=Path,
        required=True,
        help="Path to a local clone of https://github.com/apple/ml-sharp",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to SHARP .pt checkpoint (downloads default if omitted).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("web/public/models/sharp_web_predictor.onnx"),
        help="Output ONNX path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for export (cpu recommended).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=20,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Export half-precision weights (recommended for browser WASM — ~1.2 GB sidecar).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra diagnostics.",
    )
    return parser.parse_args()


DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"


def import_sharp_modules(sharp_repo: Path):
    src_path = sharp_repo / "src"
    if not src_path.exists():
        raise FileNotFoundError(
            f"Could not find {src_path}. Clone apple/ml-sharp and pass --sharp-repo."
        )
    sys.path.insert(0, str(src_path))

    import torch  # noqa: WPS433

    from sharp.models import PredictorParams, create_predictor  # noqa: WPS433

    return torch, DEFAULT_MODEL_URL, PredictorParams, create_predictor


def load_predictor(
    torch,
    create_predictor,
    predictor_params_cls,
    checkpoint_path,
    default_url,
    device,
    verbose,
):
    predictor = create_predictor(predictor_params_cls())

    if checkpoint_path is None:
        if verbose:
            print(f"Downloading checkpoint from {default_url}")
        state_dict = torch.hub.load_state_dict_from_url(default_url, progress=True)
    else:
        if verbose:
            print(f"Loading checkpoint from {checkpoint_path}")
        try:
            state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
        except TypeError:
            state_dict = torch.load(checkpoint_path, map_location=device)

    predictor.load_state_dict(state_dict)
    predictor.eval()
    predictor.to(device)
    return predictor


def make_export_module(torch, predictor):
    class SharpPredictorExport(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.predictor = predictor

        def forward(self, image, disparity_factor):
            gaussians = self.predictor(image, disparity_factor)
            return (
                gaussians.mean_vectors,
                gaussians.singular_values,
                gaussians.quaternions,
                gaussians.colors,
                gaussians.opacities,
            )

    return SharpPredictorExport()


def consolidate_external_data(output_path: Path, preexisting_files: set[str]) -> bool:
    """Merge PyTorch-exported weight shards into a single .onnx.data sidecar."""
    import onnx  # noqa: WPS433

    model_proto = onnx.load(str(output_path), load_external_data=True)
    has_external_data = any(
        initializer.data_location == onnx.TensorProto.EXTERNAL
        for initializer in model_proto.graph.initializer
    )
    if not has_external_data:
        return False

    data_filename = output_path.name + ".data"
    onnx.save_model(
        model_proto,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_filename,
        size_threshold=1024,
    )

    keep = {output_path.name, data_filename}
    for path in output_path.parent.iterdir():
        if path.name in keep or path.name in preexisting_files:
            continue
        if path.is_file():
            path.unlink()
    return True


def export_onnx() -> None:
    args = parse_args()
    sharp_repo = args.sharp_repo.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve() if args.checkpoint else None

    torch, default_url, predictor_params_cls, create_predictor = import_sharp_modules(sharp_repo)

    device = torch.device(args.device)
    predictor = load_predictor(
        torch=torch,
        create_predictor=create_predictor,
        predictor_params_cls=predictor_params_cls,
        checkpoint_path=checkpoint_path,
        default_url=default_url,
        device=device,
        verbose=args.verbose,
    )
    export_module = make_export_module(torch, predictor).eval().to(device)
    if args.fp16:
        export_module = export_module.half()
        image_dtype = torch.float16
    else:
        image_dtype = torch.float32

    dummy_image = torch.rand((1, 3, 1536, 1536), dtype=image_dtype, device=device)
    dummy_disparity_factor = torch.tensor([1.0], dtype=image_dtype, device=device)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.verbose:
        print(f"Exporting ONNX to {output_path}")

    preexisting_files = (
        {p.name for p in output_path.parent.iterdir()} if output_path.parent.exists() else set()
    )

    with torch.no_grad():
        torch.onnx.export(
            export_module,
            (dummy_image, dummy_disparity_factor),
            str(output_path),
            export_params=True,
            do_constant_folding=True,
            opset_version=args.opset,
            input_names=["image", "disparity_factor"],
            output_names=[
                "mean_vectors_ndc",
                "singular_values_ndc",
                "quaternions_ndc",
                "colors",
                "opacities",
            ],
        )

    has_external_data = consolidate_external_data(output_path, preexisting_files)

    print(f"Wrote {output_path}")
    if has_external_data:
        data_path = output_path.parent / (output_path.name + ".data")
        print(f"Wrote external tensor data {data_path}")
        print("Both files must be served together for web inference.")
    print("\nNext: cd web && npm install && npm run dev")


if __name__ == "__main__":
    export_onnx()
