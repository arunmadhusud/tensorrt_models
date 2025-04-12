import torch
import open_clip
import numpy as np
import argparse
from utils import measure_torch_time,measure_trt_time

import sys
import os
# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from trt_backend import BaseEngine


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model and tokenizer for ViT-L-14
    model_name = "ViT-L-14"
    pretrained = "laion2b_s32b_b82k"  # Recommended for ViT-L-14
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained
    )
    model = model.to(device).eval()  # Set model to evaluation mode

    tokenizer = open_clip.get_tokenizer(model_name)

    # Dummy Inputs
    dummy_image_fp32 = torch.randn(1, 3, 224, 224).to(device)
    dummy_text_tokens = torch.randint(0, 49408, (1, 77)).to(device)  # default tokenizer max_len is 77

    # Convert to numpy
    image_np = dummy_image_fp32.cpu().numpy().astype(np.float32)
    text_np = dummy_text_tokens.cpu().numpy().astype(np.int64)


    # PyTorch timing
    print("=== FP32 PyTorch Benchmarks ===")
    image_ms = measure_torch_time(lambda: model.encode_image(dummy_image_fp32))
    text_ms = measure_torch_time(lambda: model.encode_text(dummy_text_tokens))
    print(f"encode_image (fp32): {image_ms:.2f} ms")
    print(f"encode_text  (fp32): {text_ms:.2f} ms")

    # TensorRT timing
    print("\n=== TensorRT Benchmarks ===")
    image_trt = BaseEngine(args.trt_image_path)
    text_trt = BaseEngine(args.trt_text_path)
    print(f"image_encoder: {measure_trt_time(image_trt, image_np):.2f} ms")
    print(f"text_encoder : {measure_trt_time(text_trt, text_np):.2f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trt_image_path', type=str, default="clip_image_encoder.trt")
    parser.add_argument('--trt_text_path', type=str, default="clip_text_encoder.trt")
    main(parser.parse_args())