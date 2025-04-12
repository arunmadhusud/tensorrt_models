# CLIP_TRT

This repository contains code for converting CLIP models to TensorRT format for faster inference. It uses the open_clip implementation of the CLIP model architecture.

## Tested Environment

- TensorRT 10.9.0.34
- CUDA 12.4
- PyTorch 2.6.0

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### FP32 Conversion
```bash
python3 clip2trt.py --opt_batch_size 1 --max_batch_size 3
```

### FP16 Conversion
```bash
python3 clip2trt.py --opt_batch_size 1 --max_batch_size 3 --image_fp16 --text_fp16
```

## Performance Results

| Model | Encoder | PyTorch FP32 (ms) | TensorRT FP32 (ms) | TensorRT FP16 (ms) |
|:-----:|:-------:|:-----------------:|:------------------:|:-------------------:|
| CLIP<br>ViT-L/14 | Image | 67.39 | 54.50 | 12.00 |
| CLIP<br>ViT-L/14 | Text | 19.88 | 6.52 | 2.09 |


## Acknowledgements

- [Open CLIP](https://github.com/mlfoundations/open_clip) - The implementation of CLIP used in this project