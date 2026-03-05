import os
import re
import json
import random
from collections import defaultdict, deque

DATASET_ROOT = r"D:\datasets"
COMP = "c23"
SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

orig_dir = os.path.join(DATASET_ROOT, "original_sequences", "youtube", COMP, "videos")
manip_root = os.path.join(DATASET_ROOT, "manipulated_sequences")

re_one = re.compile(r"^(\d{3})\.mp4$", re.I)
re_two = re.compile(r"^(\d{3})_(\d{3})\.mp4$", re.I)

def list_mp4(folder):
    out = []
    for fn in os.listdir(folder):
        if fn.lower().endswith(".mp4"):
            out.append(fn)
    return out

# 1) 收集所有 original ids
orig_files = list_mp4(orig_dir)
orig_ids = set()
for fn in orig_files:
    m = re_one.match(fn)
    if m:
        orig_ids.add(m.group(1))

# 2) 构建 graph（节点：id；边：出现在同一个 fake 文件名里的两个 id）
g = defaultdict(set)

# 先把所有 original id 加进图（孤立点也保留）
for vid in orig_ids:
    g[vid]  # ensure exists

# 扫 manipulated 各方法目录（Deepfakes/Face2Face/...）
manip_methods = []
for name in os.listdir(manip_root):
    p = os.path.join(manip_root, name, COMP, "videos")
    if os.path.isdir(p):
        manip_methods.append((name, p))

unknown_name_files = []  # 不符合命名规则的文件，单独记录，避免乱处理

for method, vdir in manip_methods:
    for fn in list_mp4(vdir):
        m2 = re_two.match(fn)
        if m2:
            a, b = m2.group(1), m2.group(2)
            g[a].add(b)
            g[b].add(a)
        else:
            m1 = re_one.match(fn)
            if m1:
                # 有些方法可能是单 id 命名（比如 000.mp4），我们只记录节点存在
                a = m1.group(1)
                g[a]  # ensure exists
            else:
                unknown_name_files.append(os.path.join(vdir, fn))

# 3) 求连通分量
visited = set()
components = []

for node in g.keys():
    if node in visited:
        continue
    q = deque([node])
    visited.add(node)
    comp = []
    while q:
        x = q.popleft()
        comp.append(x)
        for y in g[x]:
            if y not in visited:
                visited.add(y)
                q.append(y)
    components.append(comp)

# 4) 按连通分量划分（以分量大小为权重更均衡）
random.seed(SEED)
random.shuffle(components)

total_nodes = sum(len(c) for c in components)
train_target = total_nodes * TRAIN_RATIO
val_target = total_nodes * VAL_RATIO

split_of_id = {}
cnt_train = cnt_val = cnt_test = 0

for comp in components:
    if cnt_train < train_target:
        split = "train"
        cnt_train += len(comp)
    elif cnt_val < val_target:
        split = "val"
        cnt_val += len(comp)
    else:
        split = "test"
        cnt_test += len(comp)
    for vid in comp:
        split_of_id[vid] = split

# 5) 给每个视频（original + manipulated）打 split 标签
video_split_map = {}

# original
for fn in orig_files:
    m = re_one.match(fn)
    if not m:
        continue
    vid = m.group(1)
    video_split_map[os.path.join(orig_dir, fn)] = split_of_id.get(vid, "unknown")

# manipulated
for method, vdir in manip_methods:
    for fn in list_mp4(vdir):
        path = os.path.join(vdir, fn)
        m2 = re_two.match(fn)
        if m2:
            a, b = m2.group(1), m2.group(2)
            sa, sb = split_of_id.get(a), split_of_id.get(b)
            # 在连通分量划分下，理论上 sa==sb；如果不相等就标记冲突
            if sa is None or sb is None:
                video_split_map[path] = "unknown"
            elif sa != sb:
                video_split_map[path] = "conflict"
            else:
                video_split_map[path] = sa
        else:
            m1 = re_one.match(fn)
            if m1:
                a = m1.group(1)
                video_split_map[path] = split_of_id.get(a, "unknown")
            else:
                video_split_map[path] = "unknown_name"

# 6) 输出结果
out = {
    "seed": SEED,
    "comp": COMP,
    "split_counts_ids": {
        "train": sum(1 for v in split_of_id.values() if v == "train"),
        "val": sum(1 for v in split_of_id.values() if v == "val"),
        "test": sum(1 for v in split_of_id.values() if v == "test"),
    },
    "unknown_name_files": unknown_name_files,
}

with open("split_of_id.json", "w", encoding="utf-8") as f:
    json.dump(split_of_id, f, indent=2)

with open("video_split_map.json", "w", encoding="utf-8") as f:
    json.dump(video_split_map, f, indent=2)

with open("summary.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)

print("Done. Wrote split_of_id.json, video_split_map.json, summary.json")
print("Unknown filename pattern count:", len(unknown_name_files))
print("Conflict count:", sum(1 for v in video_split_map.values() if v == "conflict"))