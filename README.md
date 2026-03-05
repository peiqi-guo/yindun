# Deepfake Risk Demo（Android + ONNX/NCNN 工具链）

> 作用：本项目提供一个可运行的 Android Demo，用悬浮球控制屏幕采集与深伪风险预警。  
> 书写思路：把“模型导出/校验/转换”与“端上推理与交互”拆成两条可复现链路，便于部署与排障。

---

## 1. 项目结构

- `app/`：Android 应用（Java）
  - `MainActivity`：权限引导、设置项（防误触/诊断开关）
  - `ProjectionPermissionActivity`：请求系统录屏授权
  - `ScreenCaptureService`：前台服务 + MediaProjection + 推理调度
  - `FloatingOverlayManager`：悬浮球与状态面板
  - `infer/`：预处理、NCNN 推理、推理 worker
- `export_onnx.py`：PyTorch -> ONNX
- `check_onnx.py`：ONNX 与 PyTorch 数值一致性校验
- `tools/ncnn/convert_onnx_to_ncnn.ps1`：ONNX -> NCNN 转换并拷贝到 Android assets

---

## 2. 环境要求

### Android 侧
- Android Studio Iguana+（或同级）
- JDK 17
- `minSdk=26`, `targetSdk=34`
- 设备支持 MediaProjection（Android 8+）

### Python 侧（导出/校验）
- Python 3.9+
- 依赖：

```bash
pip install torch torchvision onnx onnxruntime
```

### NCNN 工具（Windows）
- `onnx2ncnn.exe`
- （可选）`ncnnoptimize.exe`

---

## 3. 模型导出与校验（PyTorch -> ONNX）

### 3.1 导出 ONNX

```bash
python export_onnx.py \
  --checkpoint checkpoints/best_small.pt \
  --output artifacts/mobilenetv3_small_deepfake.onnx \
  --opset 13
```

导出约定：
- 输入名：`input`
- 输出名：`logits`
- 输入 shape：`(1, 3, 224, 224)`

### 3.2 一致性校验

```bash
python check_onnx.py \
  --checkpoint checkpoints/best_small.pt \
  --onnx artifacts/mobilenetv3_small_deepfake.onnx
```

校验会打印 `max_abs_diff`，并断言 `< 1e-3`。

---

## 4. ONNX 转 NCNN 并部署到 Android assets

在 Windows PowerShell 中：

```powershell
powershell -ExecutionPolicy Bypass -File "tools/ncnn/convert_onnx_to_ncnn.ps1" `
  -onnx_path "artifacts/mobilenetv3_small_deepfake.onnx" `
  -onnx2ncnn_path "C:\Program Files\ncnn\bin\onnx2ncnn.exe" `
  -ncnnoptimize_path "C:\Program Files\ncnn\bin\ncnnoptimize.exe" `
  -out_dir "artifacts\ncnn" `
  -android_assets_dir "app\src\main\assets"
```

最终会得到：
- `app/src/main/assets/ncnn/mobilenetv3_small.param`
- `app/src/main/assets/ncnn/mobilenetv3_small.bin`

---

## 5. Android 端部署步骤

1. 把 NCNN Android AAR 放到：
   - `app/libs/ncnn-android.aar`
2. 确认 `app/build.gradle` 已包含：
   - `implementation files('libs/ncnn-android.aar')`
3. 确认 assets 下已有模型：
   - `app/src/main/assets/ncnn/mobilenetv3_small.param`
   - `app/src/main/assets/ncnn/mobilenetv3_small.bin`
4. 用 Android Studio Sync + Run 到真机。

---

## 6. App 使用说明

1. 首次启动：授予悬浮窗权限。  
2. Android 13+：允许通知权限（拒绝也可运行，但通知展示可能受限）。  
3. 点击“显示悬浮球”。  
4. 悬浮球：
   - 单击（防误触开启时需双击确认）开始/停止检测
   - 长按打开状态面板（SAFE/SUSPICIOUS/DANGEROUS、s、p_fake）

---

## 7. 输入预处理与输出解释

端上预处理与训练一致：
- RGB
- resize 到 `224x224`
- float32
- normalize：
  - `mean=(0.485, 0.456, 0.406)`
  - `std=(0.229, 0.224, 0.225)`
- 布局：CHW（`1x3x224x224`）

模型输出：`logit`，端上计算：

```text
p_fake = sigmoid(logit)
```

---

## 8. 性能与稳定性（已做）

- `ImageReader.acquireLatestImage()` + `finally image.close()`
- 仅保留最新帧（`AtomicReference`）
- 推理固定 5fps，无新帧直接跳过
- ROI 扩框20% + 边界检查
- 复用 224 输入 Bitmap/Canvas/Rect，减少 GC
- 5秒诊断日志（fps、推理耗时、内存）可通过开关启用
- 屏幕尺寸变化自动重建 `VirtualDisplay`

---

## 9. 常见问题排查

### 9.1 启动就报找不到 NCNN 类
- 检查 `app/libs/ncnn-android.aar` 是否存在
- 检查 Gradle 依赖是否已添加并同步成功

### 9.2 提示模型加载失败
- 检查 assets 模型路径必须是：
  - `assets/ncnn/mobilenetv3_small.param`
  - `assets/ncnn/mobilenetv3_small.bin`

### 9.3 无法开始录屏
- 必须通过系统授权弹窗同意后才能启动检测
- 检查前台服务权限与通知权限

### 9.4 悬浮球不显示
- 检查悬浮窗权限是否授予

---

## 10. 二次开发建议

- 替换 `FaceBox` 逻辑为真实人脸检测
- 在 `DecisionEngine` 中按业务场景调整阈值/窗口
- 增加埋点上报和异常分级日志
- 引入真实端侧 benchmark（冷启动/稳态内存/帧时延）
