import os
import argparse
import math
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large

# --- Optional: mediapipe face detection (fast) ---
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False


# -------------------------
# Model
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
        raise ValueError("model must be 'small' or 'large'")


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# -------------------------
# Preprocess (match training)
# -------------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_bgr_to_tensor(bgr: np.ndarray, img_size: int) -> torch.Tensor:
    # BGR -> RGB
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    # HWC -> CHW
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x).unsqueeze(0)  # (1,3,H,W)


def expand_bbox(x1, y1, x2, y2, w, h, scale=0.2):
    bw = x2 - x1
    bh = y2 - y1
    x1 = int(max(0, x1 - bw * scale))
    y1 = int(max(0, y1 - bh * scale))
    x2 = int(min(w, x2 + bw * scale))
    y2 = int(min(h, y2 + bh * scale))
    return x1, y1, x2, y2


# -------------------------
# Face detection (MediaPipe) + bbox reuse
# -------------------------
class FaceDetector:
    def __init__(self, min_conf=0.6, model_selection=1):
        self.use_mp = MP_AVAILABLE
        self.min_conf = min_conf
        self.model_selection = model_selection
        self.mp_face = None
        if self.use_mp:
            self.mp_face = mp.solutions.face_detection.FaceDetection(
                model_selection=model_selection,
                min_detection_confidence=min_conf
            )
        else:
            # Fallback: OpenCV Haar (weaker, but no extra deps)
            self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect_largest(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        if self.use_mp:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = self.mp_face.process(rgb)
            if not res.detections:
                return None
            best = None
            best_area = 0
            for det in res.detections:
                bbox = det.location_data.relative_bounding_box
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)
                x2 = x1 + bw
                y2 = y1 + bh
                area = max(0, bw) * max(0, bh)
                if area > best_area:
                    best_area = area
                    best = (x1, y1, x2, y2)
            return best
        else:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            if len(faces) == 0:
                return None
            # pick largest
            x, y, bw, bh = max(faces, key=lambda f: f[2]*f[3])
            return (x, y, x+bw, y+bh)

    def close(self):
        if self.use_mp and self.mp_face is not None:
            self.mp_face.close()


# -------------------------
# Aggregation + early exit
# -------------------------
def aggregate(scores: list, mode: str, topk: float) -> float:
    if len(scores) == 0:
        return float("nan")
    s = np.array(scores, dtype=np.float32)
    if mode == "mean":
        return float(s.mean())
    if mode == "median":
        return float(np.median(s))
    if mode == "topk":
        k = max(1, int(len(s) * topk))
        ss = np.sort(s)[-k:]
        return float(ss.mean())
    raise ValueError("agg must be mean/median/topk")


def should_early_stop(scores: list, min_frames: int, thr: float, margin: float) -> bool:
    """
    简单早停：平均分数已经远离阈值 enough margin 且已积累 min_frames。
    """
    if len(scores) < min_frames:
        return False
    m = float(np.mean(scores))
    return (m >= thr + margin) or (m <= thr - margin)


# -------------------------
# Main video predict
# -------------------------
@torch.no_grad()
def predict_video(
    video_path: str,
    model: nn.Module,
    device: str,
    img_size: int = 224,
    target_fps: int = 10,
    max_frames: int = 120,         # 最多推理多少帧（控制时延）
    min_frames: int = 30,          # 早停至少看多少帧
    detect_every: int = 10,        # 每 N 个采样帧做一次人脸检测
    agg: str = "mean",             # mean/median/topk
    topk: float = 0.2,             # agg=topk 时取最高 20%
    thr: float = 0.5,              # 判别阈值（视频分数）
    early_margin: float = 0.15,    # 早停置信边界
    min_face_conf: float = 0.6,
) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"ok": False, "error": "cannot_open_video", "video": video_path}

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if not orig_fps or orig_fps <= 0:
        orig_fps = 30.0
    frame_interval = max(int(round(orig_fps / target_fps)), 1)

    detector = FaceDetector(min_conf=min_face_conf)
    last_bbox = None
    scores = []
    sampled = 0
    total_read = 0
    no_face_count = 0

    model.eval()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_read += 1
        if (total_read - 1) % frame_interval != 0:
            continue

        # choose to detect or reuse bbox
        need_detect = (last_bbox is None) or (sampled % detect_every == 0)
        if need_detect:
            bb = detector.detect_largest(frame)
            last_bbox = bb if bb is not None else None

        if last_bbox is None:
            no_face_count += 1
            sampled += 1
            if sampled >= max_frames:
                break
            continue

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = last_bbox
        x1, y1, x2, y2 = expand_bbox(x1, y1, x2, y2, w, h, scale=0.2)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            last_bbox = None
            sampled += 1
            continue

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            last_bbox = None
            sampled += 1
            continue

        x = preprocess_bgr_to_tensor(crop, img_size).to(device)
        logit = model(x).item()
        p = sigmoid(logit)
        scores.append(p)

        sampled += 1

        # early stop
        if should_early_stop(scores, min_frames=min_frames, thr=thr, margin=early_margin):
            break

        if sampled >= max_frames:
            break

    cap.release()
    detector.close()

    p_video = aggregate(scores, mode=agg, topk=topk)
    if p_video != p_video:
        return {
            "ok": False,
            "error": "no_valid_face_frames",
            "video": video_path,
            "sampled_frames": sampled,
            "no_face_frames": no_face_count,
        }

    pred = 1 if p_video >= thr else 0
    return {
        "ok": True,
        "video": video_path,
        "p_video": float(p_video),
        "pred_label": int(pred),
        "pred_text": "FAKE" if pred == 1 else "REAL",
        "used_frames": len(scores),
        "sampled_frames": sampled,
        "no_face_frames": no_face_count,
        "agg": agg,
        "thr": thr,
        "early_stopped": (len(scores) < max_frames and sampled < max_frames),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True, help="e.g. checkpoints/best_small.pt")
    ap.add_argument("--model", type=str, default="small", choices=["small", "large"])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--img_size", type=int, default=224)

    ap.add_argument("--target_fps", type=int, default=10, help="sample fps from video")
    ap.add_argument("--max_frames", type=int, default=120, help="max sampled frames to infer")
    ap.add_argument("--min_frames", type=int, default=30, help="min frames before early-stop allowed")
    ap.add_argument("--detect_every", type=int, default=10, help="run face detection every N sampled frames")

    ap.add_argument("--agg", type=str, default="mean", choices=["mean", "median", "topk"])
    ap.add_argument("--topk", type=float, default=0.2, help="topk ratio when agg=topk")
    ap.add_argument("--thr", type=float, default=0.5, help="video-level threshold")
    ap.add_argument("--early_margin", type=float, default=0.15, help="early-stop margin around thr")
    ap.add_argument("--min_face_conf", type=float, default=0.6)

    args = ap.parse_args()

    model = build_model(args.model).to(args.device)
    state = torch.load(args.ckpt, map_location=args.device)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(sd, strict=True)

    out = predict_video(
        video_path=args.video,
        model=model,
        device=args.device,
        img_size=args.img_size,
        target_fps=args.target_fps,
        max_frames=args.max_frames,
        min_frames=args.min_frames,
        detect_every=args.detect_every,
        agg=args.agg,
        topk=args.topk,
        thr=args.thr,
        early_margin=args.early_margin,
        min_face_conf=args.min_face_conf,
    )

    if not out["ok"]:
        print("FAILED:", out)
        return

    print(f"Video: {out['video']}")
    print(f"Pred:  {out['pred_text']}  (p_video={out['p_video']:.4f}, thr={out['thr']})")
    print(f"Frames used: {out['used_frames']} (sampled={out['sampled_frames']}, no_face={out['no_face_frames']})")
    print(f"Aggregation: {out['agg']}  early_stopped={out['early_stopped']}")


if __name__ == "__main__":
    main()