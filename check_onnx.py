#!/usr/bin/env python3
"""
作用：校验 ONNX 与 PyTorch 在同一随机输入上的输出一致性。
书写思路：
1) 固定 seed 保证可复现；
2) 同输入分别跑 PyTorch 和 ONNXRuntime；
3) 打印并断言 max_abs_diff < 1e-3。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from export_onnx import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PyTorch and ONNX outputs")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/best_small.pt"),
        help="Path to .pt checkpoint",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path("artifacts/mobilenetv3_small_deepfake.onnx"),
        help="Path to exported ONNX",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = load_model(args.checkpoint, device)

    x = np.random.randn(1, 3, 224, 224).astype(np.float32)

    with torch.no_grad():
        torch_out = model(torch.from_numpy(x).to(device)).cpu().numpy()

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    onnx_out = session.run(["logits"], {"input": x})[0]

    max_abs_diff = np.max(np.abs(torch_out - onnx_out))
    print(f"max_abs_diff = {max_abs_diff:.8f}")

    assert max_abs_diff < 1e-3, f"ONNX mismatch too large: {max_abs_diff}"
    print("[OK] ONNX output matches PyTorch within tolerance < 1e-3")


if __name__ == "__main__":
    main()
