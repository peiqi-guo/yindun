# ONNX → NCNN Conversion (Windows / PowerShell)

本目录提供可复现的 ONNX 到 NCNN 转换脚本：

- 脚本：`tools/ncnn/convert_onnx_to_ncnn.ps1`
- 输入 ONNX（默认建议）：`artifacts/mobilenetv3_small_deepfake.onnx`
- 输出 NCNN：
  - `mobilenetv3_small.param`
  - `mobilenetv3_small.bin`
- 并自动复制到 Android assets：`<android_assets_dir>/ncnn/`

---

## 1) 脚本参数

```powershell
-onnx_path            # 必填，ONNX 文件路径
-onnx2ncnn_path       # 必填，onnx2ncnn.exe 路径
-ncnnoptimize_path    # 可选，ncnnoptimize.exe 路径
-out_dir              # 必填，中间产物/最终 NCNN 文件输出目录
-android_assets_dir   # 必填，Android assets 根目录（如 app/src/main/assets）
```

---

## 2) 运行示例

> 建议在仓库根目录执行。Windows 路径包含空格时，请务必给路径加双引号。

### 2.1 仅转换（不做 optimize）

```powershell
powershell -ExecutionPolicy Bypass -File "tools/ncnn/convert_onnx_to_ncnn.ps1" \
  -onnx_path "artifacts/mobilenetv3_small_deepfake.onnx" \
  -onnx2ncnn_path "D:\tools\ncnn\bin\onnx2ncnn.exe" \
  -out_dir "artifacts\ncnn" \
  -android_assets_dir "app\src\main\assets"
```

### 2.2 转换 + optimize

```powershell
powershell -ExecutionPolicy Bypass -File "tools/ncnn/convert_onnx_to_ncnn.ps1" \
  -onnx_path "artifacts/mobilenetv3_small_deepfake.onnx" \
  -onnx2ncnn_path "D:\tools\ncnn\bin\onnx2ncnn.exe" \
  -ncnnoptimize_path "D:\tools\ncnn\bin\ncnnoptimize.exe" \
  -out_dir "artifacts\ncnn" \
  -android_assets_dir "app\src\main\assets"
```

### 2.3 带空格路径示例

```powershell
powershell -ExecutionPolicy Bypass -File "tools/ncnn/convert_onnx_to_ncnn.ps1" \
  -onnx_path "C:\Users\me\Deepfake Project\artifacts\mobilenetv3_small_deepfake.onnx" \
  -onnx2ncnn_path "C:\Program Files\ncnn\bin\onnx2ncnn.exe" \
  -ncnnoptimize_path "C:\Program Files\ncnn\bin\ncnnoptimize.exe" \
  -out_dir "C:\Users\me\Deepfake Project\artifacts\ncnn" \
  -android_assets_dir "C:\Users\me\Deepfake Project\app\src\main\assets"
```

---

## 3) 输出与 Android assets 放置规则

脚本会确保以下目录存在（不存在则自动创建）：

1. `-out_dir`
2. `-android_assets_dir`
3. `-android_assets_dir/ncnn`

最终固定拷贝到：

- `<android_assets_dir>/ncnn/mobilenetv3_small.param`
- `<android_assets_dir>/ncnn/mobilenetv3_small.bin`

> 即使使用 `ncnnoptimize` 生成了 `_opt` 文件，脚本也会统一复制为固定文件名，便于 Android 端固定加载路径。

---

## 4) 常见错误排查

### 4.1 找不到 exe

症状：
- `onnx2ncnn executable not found`
- `ncnnoptimize executable not found`

排查：
- 确认 exe 路径正确，尤其是 `.exe` 后缀。
- 路径含空格时用双引号。
- 手动运行 `"D:\...\onnx2ncnn.exe"` 看是否能启动。

### 4.2 权限问题 / 脚本无法执行

症状：
- PowerShell 提示脚本执行被禁止。

处理：
- 使用本 README 示例命令里的 `-ExecutionPolicy Bypass`。
- 或管理员 PowerShell 临时执行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

### 4.3 路径问题（相对路径/当前目录不一致）

症状：
- ONNX 文件找不到，或输出目录不在预期位置。

排查：
- 在仓库根目录执行脚本，或使用绝对路径。
- 检查 `-onnx_path` 指向真实文件：

```powershell
Test-Path "artifacts/mobilenetv3_small_deepfake.onnx"
```

### 4.4 onnx2ncnn / ncnnoptimize 执行失败

症状：
- 脚本报 `failed with exit code`。

排查：
- 先单独命令行运行对应 exe，确认版本/依赖完整。
- 检查 ONNX 是否损坏，可先用 `onnxruntime` 加载验证。
- 若 optimize 失败，可先不传 `-ncnnoptimize_path`，仅做基础转换。
