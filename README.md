# Deepfake Training Utilities

## Export

### 1) 安装依赖

```bash
pip install torch torchvision onnx onnxruntime
```

### 2) 导出 ONNX（固定输入形状 1x3x224x224）

```bash
python export_onnx.py \
  --checkpoint checkpoints/best_small.pt \
  --output artifacts/mobilenetv3_small_deepfake.onnx \
  --opset 13
```

说明：
- 模型结构：`mobilenet_v3_small` + 1 维输出分类头（logit）
- ONNX 输入名：`input`
- ONNX 输出名：`logits`
- 输入 shape 固定为 `(1, 3, 224, 224)`（NCHW, float32）

### 3) 校验 ONNX 与 PyTorch 一致性

```bash
python check_onnx.py \
  --checkpoint checkpoints/best_small.pt \
  --onnx artifacts/mobilenetv3_small_deepfake.onnx
```

脚本会使用同一随机输入分别跑 PyTorch 和 ONNX Runtime，打印 `max_abs_diff` 并断言 `< 1e-3`。

### 4) 输入预处理与端上后处理

模型训练/推理输入预处理：
- RGB
- resize/crop 到 `224x224`
- 转 `float32`
- 归一化：
  - `mean=(0.485, 0.456, 0.406)`
  - `std=(0.229, 0.224, 0.225)`
- 张量布局：`NCHW`，即 `(1, 3, 224, 224)`

模型输出为 **1 个 logit**，端上概率按以下方式计算：

```python
p_fake = sigmoid(logit)
```
