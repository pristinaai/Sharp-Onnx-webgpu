#!/usr/bin/env python3
"""Validate exported SHARP ONNX model against PyTorch reference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sharp-repo", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-4)
    return parser.parse_args()


DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"


def main() -> None:
    args = parse_args()
    sharp_repo = args.sharp_repo.expanduser().resolve()
    src_path = sharp_repo / "src"
    sys.path.insert(0, str(src_path))

    import numpy as np
    import onnxruntime as ort
    import torch

    from sharp.models import PredictorParams, create_predictor

    device = torch.device("cpu")
    predictor = create_predictor(PredictorParams())

    if args.checkpoint:
        state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    else:
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)

    predictor.load_state_dict(state_dict)
    predictor.eval()

    image = torch.rand(1, 3, 1536, 1536)
    disparity_factor = torch.tensor([0.85])

    with torch.no_grad():
        ref = predictor(image, disparity_factor)

    session = ort.InferenceSession(
        str(args.onnx),
        providers=["CPUExecutionProvider"],
    )
    outputs = session.run(
        None,
        {
            "image": image.numpy(),
            "disparity_factor": disparity_factor.numpy(),
        },
    )

    names = [
        "mean_vectors_ndc",
        "singular_values_ndc",
        "quaternions_ndc",
        "colors",
        "opacities",
    ]
    ref_tensors = [
        ref.mean_vectors.numpy(),
        ref.singular_values.numpy(),
        ref.quaternions.numpy(),
        ref.colors.numpy(),
        ref.opacities.numpy(),
    ]

    ok = True
    for name, onnx_out, ref_out in zip(names, outputs, ref_tensors, strict=True):
        max_diff = np.max(np.abs(onnx_out - ref_out))
        close = np.allclose(onnx_out, ref_out, rtol=args.rtol, atol=args.atol)
        status = "OK" if close else "FAIL"
        print(f"{name}: max_diff={max_diff:.6e} [{status}]")
        ok = ok and close

    if ok:
        print("\nValidation passed.")
    else:
        print("\nValidation failed — check export or tolerances.")
        sys.exit(1)


if __name__ == "__main__":
    main()
