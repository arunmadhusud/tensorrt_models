# COCA_TRT

This repository contains code for converting [Contrastive Captioners](https://arxiv.org/pdf/2205.01917) (CoCa) model to TensorRT format for faster inference. It uses the open_clip implementation of the CoCa model architecture.

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

CoCa Image Encoder

```bash
python3 ie2trt.py --opt_batch_size 1 --max_batch_size 3
```
CoCa Text Encoder (Same as Unimodal Text Decoder described in the paper)
```bash
python3 te2trt.py --opt_batch_size 1 --max_batch_size 3
```
CoCa Text Decoder (Same as Multimodal Text Encoder described in the paper)
```bash
python3 td2trt.py --opt_batch_size 1 --max_batch_size 3
```

### FP16 Conversion
CoCa Image Encoder
```bash
python3 ie2trt.py --opt_batch_size 1 --max_batch_size 3 --image_fp16 
```
CoCa Text Encoder (Same as Unimodal Text Decoder described in the paper)
```bash
python3 te2trt.py --opt_batch_size 1 --max_batch_size 3  --text_fp16
```
CoCa Text Decoder (Same as Multimodal Text Encoder described in the paper)
```bash
python3 td2trt.py --opt_batch_size 1 --max_batch_size 3 --text_fp16
```
### For generating captions using greedy search
```bash
python3 caption_gen.py --image_path images/dog.jpg --image_encoder coca_image_encoder.trt --text_encoder coca_text_encoder.trt --text_decoder coca_text_decoder.trt
```

## Performance Results

| Model | Type | PyTorch FP32 (ms) | TensorRT FP32 (ms) | TensorRT FP16 (ms) |
|:-----:|:-------:|:-----------------:|:------------------:|:-------------------:|
| coca<br>ViT-L/14 | Image Encoder | 59.18 | 49.90 | 11.35 |
| coca<br>ViT-L/14 | Text Encoder | 10.77 | 3.72 | 1.41 |
| coca<br>ViT-L/14 | Text Decoder | 17.53 | 11.64 | 7.21 |


## Acknowledgements

- [Open CLIP](https://github.com/mlfoundations/open_clip) - The implementation of CoCa model used in this project