# train_mobilenetv3_faces_full_earlystop.py
# 训练 MobileNetV3 (small/large) 做 FF++ 空间分支（基于 preprocess_faces.py 输出的人脸帧）
# 输出：
# - checkpoints/best_{small|large}.pt
# - training_log.csv
# - curve_train_loss.png, curve_val_auc.png
# - test_roc.png, test_confusion.png
#
# 用法：
# 1) 修改 DATA_ROOT 为你的预处理根目录（包含 train/val/test/faces）
# 2) python train_mobilenetv3_faces_full_earlystop.py

import os
import random
from typing import List

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large

from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score
)

import pandas as pd
import matplotlib.pyplot as plt


# =======================
# 配置：只需要改 DATA_ROOT
# =======================
DATA_ROOT = r"D:\datasets\FF++_preprocessed"  # preprocess_faces.py 输出根目录
MODEL_NAME = "small"  # "small" or "large"
IMG_SIZE = 224        # 224 更快；预处理256也可训练时resize
VAL_FRAMES_PER_VIDEO = 20
TEST_FRAMES_PER_VIDEO = 40

BATCH_SIZE = 16
EPOCHS = 30           # 上限；早停可能提前结束
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42
NUM_WORKERS = 2       # Windows 建议 0 或 2

# Early Stopping（按 val AUC）
PATIENCE = 4          # 连续多少个epoch没提升就停
MIN_DELTA = 1e-4      # 提升小于该阈值不算提升

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def label_from_path(path: str) -> int:
    p = path.replace("/", "\\")
    if "\\original_sequences\\" in p:
        return 0
    if "\\manipulated_sequences\\" in p:
        return 1
    raise ValueError(f"Cannot infer label from path: {path}")


def is_video_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    for fn in os.listdir(path):
        if fn.lower().endswith((".jpg", ".jpeg", ".png")):
            return True
    return False


def list_video_dirs(split_faces_root: str) -> List[str]:
    out = []
    for root, _, _ in os.walk(split_faces_root):
        if is_video_dir(root):
            out.append(root)
    out.sort()
    return out


def list_frames(video_dir: str) -> List[str]:
    frames = []
    for fn in os.listdir(video_dir):
        if fn.lower().endswith((".jpg", ".jpeg", ".png")):
            frames.append(os.path.join(video_dir, fn))
    frames.sort()
    return frames


class VideoFrameTrainDataset(Dataset):
    """
    每个样本对应一个视频文件夹
    每次 __getitem__ 随机抽一帧训练：避免长视频贡献过多
    """
    def __init__(self, video_dirs: List[str], transform):
        self.transform = transform
        self.video_frames = []
        self.labels = []

        for vd in video_dirs:
            fr = list_frames(vd)
            if len(fr) == 0:
                continue
            self.video_frames.append(fr)
            self.labels.append(label_from_path(vd))

    def __len__(self):
        return len(self.video_frames)

    def __getitem__(self, idx):
        frames = self.video_frames[idx]
        y = self.labels[idx]
        fp = random.choice(frames)

        from PIL import Image
        img = Image.open(fp).convert("RGB")
        x = self.transform(img)
        return x, torch.tensor(y, dtype=torch.float32)


@torch.no_grad()
def eval_video_level(model: nn.Module, video_dirs: List[str], transform, frames_per_video: int):
    """
    每个视频抽 frames_per_video 帧，平均 logit -> sigmoid 得到视频分数
    返回：auc, stats, y_true, y_score
    """
    model.eval()
    y_true, y_score = [], []

    for vd in tqdm(video_dirs, desc=f"Eval(video-level, {frames_per_video}f)"):
        label = label_from_path(vd)
        frames = list_frames(vd)
        if len(frames) == 0:
            continue

        if len(frames) <= frames_per_video:
            picked = frames
        else:
            idxs = np.linspace(0, len(frames) - 1, frames_per_video).astype(int).tolist()
            picked = [frames[i] for i in idxs]

        logits = []
        for fp in picked:
            from PIL import Image
            img = Image.open(fp).convert("RGB")
            x = transform(img).unsqueeze(0).to(DEVICE)
            logits.append(model(x).item())

        video_logit = float(np.mean(logits))
        video_prob = 1.0 / (1.0 + np.exp(-video_logit))

        y_true.append(label)
        y_score.append(video_prob)

    auc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else float("nan")
    stats = {"n_videos": len(y_true), "pos": int(sum(y_true)), "neg": int(len(y_true) - sum(y_true))}
    return auc, stats, y_true, y_score


def build_model(name: str) -> nn.Module:
    if name == "small":
        m = mobilenet_v3_small(weights="DEFAULT")
        in_dim = m.classifier[3].in_features
        m.classifier[3] = nn.Linear(in_dim, 1)
        return m
    elif name == "large":
        m = mobilenet_v3_large(weights="DEFAULT")
        in_dim = m.classifier[3].in_features
        m.classifier[3] = nn.Linear(in_dim, 1)
        return m
    else:
        raise ValueError("MODEL_NAME must be 'small' or 'large'")


def save_curves(history_csv: str):
    df = pd.read_csv(history_csv)

    plt.figure()
    plt.plot(df["epoch"], df["train_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Train Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.savefig("curve_train_loss.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(df["epoch"], df["val_auc"])
    plt.xlabel("Epoch")
    plt.ylabel("Val AUC")
    plt.title("Val AUC Curve")
    plt.grid(True, alpha=0.3)
    plt.savefig("curve_val_auc.png", dpi=200, bbox_inches="tight")
    plt.close()


def save_test_plots(y_true: List[int], y_score: List[float], test_auc: float, thr: float = 0.5):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC (AUC={test_auc:.4f})")
    plt.grid(True, alpha=0.3)
    plt.savefig("test_roc.png", dpi=200, bbox_inches="tight")
    plt.close()

    y_pred = [1 if s >= thr else 0 for s in y_score]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real(0)", "Fake(1)"])
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, values_format="d")
    acc = accuracy_score(y_true, y_pred)
    ax.set_title(f"Confusion (thr={thr:.2f}, Acc={acc:.4f})")
    plt.savefig("test_confusion.png", dpi=200, bbox_inches="tight")
    plt.close()


def main():
    set_seed(SEED)
    print("Device:", DEVICE)

    train_root = os.path.join(DATA_ROOT, "train", "faces")
    val_root = os.path.join(DATA_ROOT, "val", "faces")
    test_root = os.path.join(DATA_ROOT, "test", "faces")

    train_videos = list_video_dirs(train_root)
    val_videos = list_video_dirs(val_root)
    test_videos = list_video_dirs(test_root)

    print(f"train videos: {len(train_videos)}")
    print(f"val videos:   {len(val_videos)}")
    print(f"test videos:  {len(test_videos)}")

    if len(train_videos) == 0:
        raise RuntimeError("No train videos found. Check DATA_ROOT and preprocess output.")
    if len(val_videos) == 0:
        print("WARN: val videos is 0. Early stopping disabled (no val).")
    if len(test_videos) == 0:
        print("WARN: test videos is 0. Test plots will be skipped.")

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    ds_train = VideoFrameTrainDataset(train_videos, transform=train_tf)
    dl_train = DataLoader(
        ds_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        drop_last=False
    )

    model = build_model(MODEL_NAME).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    os.makedirs("checkpoints", exist_ok=True)
    best_ckpt = os.path.join("checkpoints", f"best_{MODEL_NAME}.pt")

    history = []
    best_val_auc = -1.0
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []

        pbar = tqdm(dl_train, desc=f"Epoch {epoch}/{EPOCHS}")
        for x, y in pbar:
            x = x.to(DEVICE)
            y = y.to(DEVICE).unsqueeze(1)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            pbar.set_postfix(loss=float(np.mean(losses)))

        train_loss = float(np.mean(losses)) if losses else float("nan")

        # val AUC
        if len(val_videos) > 0:
            val_auc, val_stats, _, _ = eval_video_level(model, val_videos, eval_tf, frames_per_video=VAL_FRAMES_PER_VIDEO)
        else:
            val_auc, val_stats = float("nan"), {"n_videos": 0, "pos": 0, "neg": 0}

        print(f"[Epoch {epoch}] train_loss={train_loss:.6f} val_auc={val_auc:.4f} stats={val_stats}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_auc": float(val_auc) if val_auc == val_auc else np.nan,
            "val_n_videos": val_stats["n_videos"],
            "val_pos": val_stats["pos"],
            "val_neg": val_stats["neg"],
        })
        pd.DataFrame(history).to_csv("training_log.csv", index=False)

        # 保存 best（按 val_auc）
        if val_auc == val_auc:  # not NaN
            if val_auc > best_val_auc + MIN_DELTA:
                best_val_auc = val_auc
                best_epoch = epoch
                no_improve = 0
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_auc": val_auc}, best_ckpt)
                print("Saved best:", best_ckpt)
            else:
                no_improve += 1
                print(f"EarlyStop: no improvement {no_improve}/{PATIENCE}")
                if no_improve >= PATIENCE:
                    print(f"EarlyStop triggered at epoch {epoch}. Best epoch={best_epoch}, best val_auc={best_val_auc:.4f}")
                    break
        else:
            # 没有 val：不做早停，训练到最后；保存最后一次
            if epoch == EPOCHS:
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_auc": val_auc}, best_ckpt)
                print("Saved last (no val):", best_ckpt)

    # 曲线
    if os.path.exists("training_log.csv"):
        save_curves("training_log.csv")
        print("Saved curves: curve_train_loss.png, curve_val_auc.png")

    # 加载 best 做 test
    if os.path.exists(best_ckpt):
        state = torch.load(best_ckpt, map_location=DEVICE)
        model.load_state_dict(state["model"])
        print("Loaded best checkpoint:", best_ckpt, "epoch:", state.get("epoch"), "val_auc:", state.get("val_auc"))

    # 测试 + 图
    if len(test_videos) > 0:
        test_auc, test_stats, y_true, y_score = eval_video_level(model, test_videos, eval_tf, frames_per_video=TEST_FRAMES_PER_VIDEO)
        print(f"[TEST] auc={test_auc:.4f} stats={test_stats}")
        save_test_plots(y_true, y_score, test_auc, thr=0.5)
        print("Saved: test_roc.png, test_confusion.png")
    else:
        print("WARN: No test videos found. Skip test plots.")


if __name__ == "__main__":
    main()