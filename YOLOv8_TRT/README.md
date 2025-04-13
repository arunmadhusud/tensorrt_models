# YOLOv8_TRT
This repository contains code for converting YOLOv8 models to TensorRT format for faster inference. For improved performance, the TensorRT engine is built with an added NMS (Non-Maximum Suppression) operation using the EfficientNMS_TRT plugin.

## Tested Environment

- TensorRT 10.9.0.34
- CUDA 12.4
- PyTorch 2.6.0
- Nvidia T4 GPU

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### FP32 Conversion

```bash
python3 yolo2trt.py --opt_batch_size 1 --max_batch_size 3
```

### FP16 Conversion

```bash
python3 yolo2trt.py --opt_batch_size 1 --max_batch_size 3 --fp16 
```
### For inference using TensorRT engine
```bash
python3 infer_trt.py -e yolov8x.trt -i images/bus.jpg
```

## Performance Results

Note about NMS: The TensorRT implementation includes EfficientNMS_TRT plugin for Non-Maximum Suppression operations directly in the TensorRT engine. This integration eliminates the need for separate post-processing steps, significantly improving overall inference time compared to the PyTorch implementation which requires separate NMS operation after the model inference, leading to longer processing times.

| Model | PyTorch FP32 (ms) (without NMS) | TensorRT FP32 (ms)(with NMS) | TensorRT FP16 (ms)(with NMS) |
|:-----:|:-----------------:|:------------------:|:-------------------:|
| Yolov8n | 8.18 | 4.3 | 2.67 |

## Acknowledgements

- [Ultralytics](https://docs.ultralytics.com/) 
- [TensorRT-For-YOLO-Series ](https://github.com/Linaom1214/TensorRT-For-YOLO-Series) 
- [yolov7](https://github.com/WongKinYiu/yolov7.git)