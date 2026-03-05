import os
import json
import argparse
from typing import List, Dict
from collections import defaultdict

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large

from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import csv


# -------------------------
# Utils
# -------------------------
def build_model(model_name: str) -> nn.Module:
    if model_name == "small":
        m = mobilenet_v3_small(weights=None)
        in_dim = m.classifier[3].in_features
        m.classifier[3] = nn.Linear(in_dim, 1)
        return m
    elif model_name == "large":
        m = mobilenet_v3_large(weights=None)
        in_dim = m.classifier[3].in_features
        m.classifier[3] = nn.Linear(in_dim, 1)
        return m
    else:
        raise ValueError("model_name must be 'small' or 'large'")


def list_frames(video_dir: str) -> List[str]:
    if not os.path.isdir(video_dir):
        return []
    xs = [
        os.path.join(video_dir, fn)
        for fn in os.listdir(video_dir)
        if fn.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    xs.sort()
    return xs


def save_roc(y_true, y_score, out_path: str, title: str):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else float("nan")
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title} (AUC={auc:.4f})")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_confusion(y_true, y_score, thr: float, out_path: str, title: str):
    y_pred = [1 if s >= thr else 0 for s in y_score]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real(0)", "Fake(1)"])
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, values_format="d")
    acc = accuracy_score(y_true, y_pred)
    ax.set_title(f"{title} (thr={thr:.2f}, Acc={acc:.4f})")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def print_report(y_true, y_score, thr: float, name: str):
    y_pred = np.array([1 if s >= thr else 0 for s in y_score], dtype=int)
    y_true = np.array(y_true, dtype=int)
    auc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else float("nan")

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    p1, r1, f11 = float(p[1]), float(r[1]), float(f1[1])

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print(f"\n== {name} ==")
    print(f"AUC(video-level): {auc:.4f}")
    print(f"thr={thr:.3f}  Acc={acc:.4f}  Precision(Fake)={p1:.4f}  Recall(Fake)={r1:.4f}  F1(Fake)={f11:.4f}")
    print(f"Confusion [tn fp; fn tp] = [{tn} {fp}; {fn} {tp}]")


def choose_threshold(y_true, y_score, strategy: str, target_recall: float):
    y_true = np.array(y_true, dtype=int)
    y_score = np.array(y_score, dtype=float)

    if strategy == "youden":
        fpr, tpr, thr = roc_curve(y_true, y_score)
        j = tpr - fpr
        k = int(np.argmax(j))
        return float(thr[k]), {"tpr": float(tpr[k]), "fpr": float(fpr[k]), "youdenJ": float(j[k])}

    thresholds = np.linspace(0.01, 0.99, 199)
    best_thr = 0.5
    best_metric = -1.0
    best_info = None

    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
        p1, r1, f11 = float(p[1]), float(r[1]), float(f1[1])

        if strategy == "f1":
            metric = f11
        elif strategy == "recall":
            if r1 < target_recall:
                continue
            metric = p1
        else:
            raise ValueError("strategy must be one of: f1 / recall / youden")

        if metric > best_metric:
            best_metric = metric
            best_thr = float(thr)
            best_info = {"precision_fake": p1, "recall_fake": r1, "f1_fake": f11}

    return best_thr, best_info


def load_ckpt_flexible(model: nn.Module, ckpt_path: str, device: str):
    state = torch.load(ckpt_path, map_location=device)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("Loaded ckpt:", ckpt_path)
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)


def infer_method(rel_video: str) -> str:
    # 用 rel_video 的顶层目录当 bucket
    rp = rel_video.replace("\\", "/")
    return rp.split("/", 1)[0] if "/" in rp else "Unknown"


@torch.no_grad()
def eval_from_labels_json(
    model: nn.Module,
    labels_json: str,
    tfm,
    device: str,
    frames_per_video: int,
):
    with open(labels_json, "r", encoding="utf-8") as f:
        m = json.load(f)

    from PIL import Image

    y_true, y_score, methods, rel_videos = [], [], [], []
    skipped_no_frames = 0

    model.eval()

    for rel_video, info in tqdm(m.items(), desc=f"Eval labels(video-level {frames_per_video}f)"):
        out_dir = info["out_dir"]
        label = int(info["label"])
        frames = list_frames(out_dir)
        if len(frames) == 0:
            skipped_no_frames += 1
            continue

        if len(frames) <= frames_per_video:
            picked = frames
        else:
            idxs = np.linspace(0, len(frames) - 1, frames_per_video).astype(int).tolist()
            picked = [frames[i] for i in idxs]

        logits = []
        for fp in picked:
            img = Image.open(fp).convert("RGB")
            x = tfm(img).unsqueeze(0).to(device)
            logits.append(model(x).item())

        prob = 1.0 / (1.0 + np.exp(-float(np.mean(logits))))
        y_true.append(label)
        y_score.append(prob)
        rel_videos.append(rel_video)
        methods.append(infer_method(info.get("rel_video", rel_video)))

    auc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else float("nan")
    stats = {
        "n_videos": len(y_true),
        "pos(fake=1)": int(sum(y_true)),
        "neg(real=0)": int(len(y_true) - sum(y_true)),
        "skipped_no_frames": int(skipped_no_frames),
    }
    return auc, stats, rel_videos, y_true, y_score, methods


def per_method_auc(y_true, y_score, methods):
    buckets = defaultdict(lambda: {"y": [], "s": []})
    for y, s, m in zip(y_true, y_score, methods):
        buckets[m]["y"].append(y)
        buckets[m]["s"].append(s)
    out = {}
    for m, d in buckets.items():
        if len(set(d["y"])) < 2:
            continue
        out[m] = float(roc_auc_score(d["y"], d["s"]))
    return dict(sorted(out.items(), key=lambda x: x[0]))


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="path to best_*.pt or finetuned_*.pt")
    ap.add_argument("--model", type=str, default="small", choices=["small", "large"])
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--frames_per_video", type=int, default=80)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # labels.json 输入
    ap.add_argument("--labels_json", type=str, required=True, help="e.g. D:/.../CelebDFv2_preprocessed/test/labels.json")

    # Threshold / reports
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--select_thr", action="store_true", help="choose threshold on THIS set (for confusion only)")
    ap.add_argument("--thr_strategy", type=str, default="f1", choices=["f1", "recall", "youden"])
    ap.add_argument("--target_recall", type=float, default=0.90)

    ap.add_argument("--per_method", action="store_true")
    ap.add_argument("--out_dir", type=str, default="eval_out_labels")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    model = build_model(args.model).to(args.device)
    load_ckpt_flexible(model, args.ckpt, args.device)
    model.eval()

    auc, stats, rel_videos, y_true, y_score, methods = eval_from_labels_json(
        model, args.labels_json, tfm, args.device, frames_per_video=args.frames_per_video
    )
    print(f"\n[Eval] auc={auc:.4f} stats={stats}")

    thr = args.threshold
    if args.select_thr:
        thr, info = choose_threshold(y_true, y_score, args.thr_strategy, args.target_recall)
        print(f"[SelectThr on THIS set] chosen_thr={thr:.3f} info={info}")

    print_report(y_true, y_score, thr, name="Report")

    save_roc(y_true, y_score, os.path.join(args.out_dir, "roc.png"), title="ROC")
    save_confusion(y_true, y_score, thr, os.path.join(args.out_dir, "confusion.png"), title="Confusion")

    if args.per_method:
        pm = per_method_auc(y_true, y_score, methods)
        if pm:
            print("\nPer-bucket AUC (top folder):")
            for k, v in pm.items():
                print(f"  {k:18s} {v:.4f}")

    out_scores = os.path.join(args.out_dir, "video_scores.csv")
    with open(out_scores, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rel_video", "label(0=real,1=fake)", "bucket", "score"])
        for rv, y, m, s in zip(rel_videos, y_true, methods, y_score):
            w.writerow([rv, y, m, f"{s:.6f}"])

    print(f"\nSaved: {out_scores}")
    print(f"Saved: {os.path.join(args.out_dir, 'roc.png')}")
    print(f"Saved: {os.path.join(args.out_dir, 'confusion.png')}")

if __name__ == "__main__":
    main()