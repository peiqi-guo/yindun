import os
import json
import cv2
import argparse
from tqdm import tqdm
import mediapipe as mp
import multiprocessing as mp_pool

mp_face = mp.solutions.face_detection


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def already_processed(out_dir, min_frames=1):
    if not os.path.isdir(out_dir):
        return False
    cnt = 0
    for fn in os.listdir(out_dir):
        if fn.lower().endswith((".jpg", ".jpeg", ".png")):
            cnt += 1
            if cnt >= min_frames:
                return True
    return False
def expand_bbox(x1, y1, x2, y2, w, h, scale=0.2):
    bw = x2 - x1
    bh = y2 - y1
    x1 = int(max(0, x1 - bw * scale))
    y1 = int(max(0, y1 - bh * scale))
    x2 = int(min(w, x2 + bw * scale))
    y2 = int(min(h, y2 + bh * scale))
    return x1, y1, x2, y2


def detect_largest_face(detector, frame_bgr):
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = detector.process(rgb)
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


def video_rel_key(src_path, dataset_root):
    # 让输出目录保持原始层级结构：original_sequences/... 或 manipulated_sequences/...
    rel = os.path.relpath(src_path, dataset_root)
    rel = rel.replace(":", "")  # 防止 Windows 盘符导致奇怪路径
    return rel


def process_one_video(args_tuple):
    (video_path, dataset_root, out_faces_root, target_fps, frame_limit,
     jpg_quality, model_selection, min_conf) = args_tuple

    # 每个进程自己创建 detector（MediaPipe 对多进程更稳）
    with mp_face.FaceDetection(model_selection=model_selection, min_detection_confidence=min_conf) as detector:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return (video_path, "open_failed")

        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if not orig_fps or orig_fps <= 0:
            orig_fps = target_fps
        frame_interval = max(int(round(orig_fps / target_fps)), 1)

        rel = video_rel_key(video_path, dataset_root)
        # 输出目录：.../faces/<rel_without_ext>/
        rel_no_ext = os.path.splitext(rel)[0]
        out_dir = os.path.join(out_faces_root, rel_no_ext)
        ensure_dir(out_dir)

        # ✅ 已经有输出就跳过（例如至少10帧才算完成，防止上次中断）
        if already_processed(out_dir, min_frames=10):
            return (video_path, "skipped")

        frame_idx = 0
        saved_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            if frame_limit > 0 and saved_idx >= frame_limit:
                break

            bbox = detect_largest_face(detector, frame)
            if bbox is None:
                frame_idx += 1
                continue

            h, w = frame.shape[:2]
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = expand_bbox(x1, y1, x2, y2, w, h, scale=0.2)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                frame_idx += 1
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                frame_idx += 1
                continue

            crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)

            save_path = os.path.join(out_dir, f"{saved_idx:05d}.jpg")
            cv2.imwrite(save_path, crop, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])

            saved_idx += 1
            frame_idx += 1

        cap.release()
        return (video_path, "ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--split_map", type=str, required=True)  # video_split_map.json
    ap.add_argument("--out_root", type=str, required=True)    # processed/<split>/
    ap.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    ap.add_argument("--target_fps", type=int, default=10)     # 空间分支推荐10；你要25也行
    ap.add_argument("--frame_limit", type=int, default=0)     # 0=不限制
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--jpg_quality", type=int, default=95)
    ap.add_argument("--model_selection", type=int, default=1)
    ap.add_argument("--min_conf", type=float, default=0.6)
    args = ap.parse_args()

    with open(args.split_map, "r", encoding="utf-8") as f:
        video_split = json.load(f)

    out_faces_root = os.path.join(args.out_root, args.split, "faces")
    ensure_dir(out_faces_root)

    # 收集本 split 视频
    video_list = [p for p, s in video_split.items() if s == args.split and p.lower().endswith(".mp4")]
    video_list.sort()

    if not video_list:
        print("No videos for split:", args.split)
        return

    tasks = [
        (vp, args.dataset_root, out_faces_root, args.target_fps, args.frame_limit,
         args.jpg_quality, args.model_selection, args.min_conf)
        for vp in video_list
    ]

    # Windows：用 spawn，多进程稳一点
    workers = max(1, args.workers)
    with mp_pool.get_context("spawn").Pool(processes=workers) as pool:
        for vp, status in tqdm(pool.imap_unordered(process_one_video, tasks), total=len(tasks)):
            if status != "ok":
                print("WARN:", status, vp)

    print("Done faces:", out_faces_root)


if __name__ == "__main__":
    main()