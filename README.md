# Face Classification - MobileNetV2 0.35

Mô hình phân loại khuôn mặt nhị phân (Face Occlusion Classification: `clear` vs `occluded`) sử dụng kiến trúc MobileNetV2 0.35 tối ưu hóa cho NPU và các thiết bị biên.

## Cấu trúc thư mục

```text
face_classification_mobilenetv2/
├── config/
│   └── config.yaml             # Cấu hình tham số mô hình, training, export
├── src/
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── dataset.py          # Xử lý dữ liệu & Data Augmentation
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py          # Tính toán Accuracy, F1-score, Logit range
│   ├── inference/
│   │   ├── __init__.py
│   │   └── classifier.py       # Wrapper chạy inference với ONNX Runtime
│   ├── models/
│   │   ├── __init__.py
│   │   └── mobilenetv2.py      # Kiến trúc MobileNetV2 0.35 backbone
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py           # Focal Loss + Logit Penalty Loss
│   │   └── trainer.py          # Vòng lặp huấn luyện, Early Stopping
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── model.py
│       ├── seed.py
│       └── training.py
├── checkpoints/
│   └── best.pth                # Checkpoint PyTorch tốt nhất
├── outputs/
│   └── onnx/
│       └── mobilenetv2_035_face_occlusion.onnx  # Mô hình ONNX
├── train.py                    # Huấn luyện mô hình
├── test.py                     # Đánh giá mô hình trên tập validation/test
├── export_onnx.py              # Xuất mô hình sang định dạng ONNX
├── inference.py                # Chạy inference nhanh trên ảnh đơn
└── requirements.txt
```

## Cài đặt môi trường

```bash
pip install -r requirements.txt
```

## Huấn luyện mô hình (Training)

```bash
python train.py --config config/config.yaml
```

## Đánh giá mô hình (Evaluation)

```bash
python test.py --config config/config.yaml
```

## Xuất ONNX (Export ONNX)

```bash
python export_onnx.py --config config/config.yaml
```

File ONNX được xuất tại `outputs/onnx/mobilenetv2_035_face_occlusion.onnx`.

## Inference ảnh đơn

```bash
python inference.py path/to/face.jpg --config config/config.yaml
```
