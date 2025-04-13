# **TensorRT-Models**  
The aim of this repository is to provide a codebase for converting various pytorch/onnx models to TensorRT format for faster inference using NVIDIA GPUs. Precision modes supported are FP32 and FP16 as of now. 

## ✅ **Supported Models**

- [x] [**Open_clip CLIP**](https://github.com/mlfoundations/open_clip) – Contrastive Language-Image Pretraining
- [x] [**Open_clip CoCa**](https://github.com/mlfoundations/open_clip) – Contrastive Captioners
- [x] [**Ultralytics YOLOv8**](https://github.com/ultralytics/ultralytics)

## Usage

Check out the model folders for usage instructions:

- [`CLIP_TRT`](CLIP_TRT/) – For CLIP conversion.  
- [`COCA_TRT`](COCA_TRT/) – For CoCa conversion.  
- [`YOLOv8_TRT`](YOLOv8/) – For YOLOv8 conversion.  

# Performance Results*

| Model | Component | PyTorch FP32 (ms) | TensorRT FP32 (ms) | TensorRT FP16 (ms) | Speedup (FP16) |
|:-----:|:---------:|:-----------------:|:------------------:|:------------------:|:--------------:|
| CLIP<br>ViT-L/14 | Image Encoder | 57.39 | 48.89 | 11.35 | 5.06× |
| CLIP<br>ViT-L/14 | Text Encoder | 10.98 | 5.71 | 1.84 | 5.97× |
| CoCa<br>ViT-L/14 | Image Encoder | 59.18 | 49.90 | 11.35 | 5.21× |
| CoCa<br>ViT-L/14 | Text Encoder | 10.77 | 3.72 | 1.41 | 7.64× |
| CoCa<br>ViT-L/14 | Text Decoder | 17.53 | 11.64 | 7.21 | 2.43× |
| YOLOv8n** | -- | 8.18 | 4.30 | 2.67 | 3.06× |

*batch size used for all measurements is 1.

**YOLOv8n PyTorch measurements do not include NMS, while TensorRT versions include integrated NMS via EfficientNMS_TRT plugin.
