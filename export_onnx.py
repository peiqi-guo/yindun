#!/usr/bin/env python3
"""
作用：将训练好的 MobileNetV3-small（二分类，1 个 logit）导出为 ONNX。
书写思路：
1) 构建与训练一致的网络头；
2) 兼容常见 checkpoint 键前缀；
3) 固定输入 shape/输入输出名，便于端上稳定对接。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small


class DeepfakeMobileNetV3Small(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = mobilenet_v3_small(weights=None)
        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_features, 1)
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def _normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove common distributed-training prefixes."""
    normalized = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        normalized[nk] = v
    return normalized


def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model", ckpt)
    state_dict = _normalize_state_dict_keys(state_dict)

    model = DeepfakeMobileNetV3Small().to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys ({len(missing)}): {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys ({len(unexpected)}): {unexpected}")

    model.eval()
    return model


def export_onnx(checkpoint: Path, output: Path, opset: int = 13) -> None:
    device = torch.device("cpu")
    model = load_model(checkpoint, device)

    output.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        str(output),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,
    )

    print(f"[OK] Exported ONNX to: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MobileNetV3-small deepfake model to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/best_small.pt"),
        help="Path to .pt checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/mobilenetv3_small_deepfake.onnx"),
        help="Output ONNX path",
    )
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_onnx(args.checkpoint, args.output, opset=args.opset)
