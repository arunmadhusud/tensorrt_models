import torch
import open_clip
import numpy as np
import argparse
import sys
import os
from utils import measure_torch_time, measure_trt_time

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from trt_backend import BaseEngine





def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CoCa model
    model, _, _ = open_clip.create_model_and_transforms(
        model_name="coca_ViT-L-14",
        pretrained="mscoco_finetuned_laion2B-s13B-b90k"
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-14")

    # Dummy Inputs
    dummy_image = torch.randn(1, 3, 224, 224).to(device)
    dummy_text_tokens = torch.randint(0, 49408, (1, 76)).to(device)
    dummy_image_embs = torch.randn(1, 255, 768).to(device)
    dummy_token_embs = torch.randn(1, 6, 768).to(device)

    print("=== PyTorch (FP32) Benchmarks ===")
    print(f"image_encoder: {measure_torch_time(lambda: model._encode_image(dummy_image, normalize=True)):.2f} ms per run")
    print(f"text_encoder: {measure_torch_time(lambda: model._encode_text(dummy_text_tokens, normalize=True)):.2f} ms per run")
    print(f"text_decoder: {measure_torch_time(lambda: model.text_decoder(dummy_image_embs, dummy_token_embs)):.2f} ms per run")

    # Convert to NumPy
    image_np = dummy_image.cpu().numpy().astype(np.float32)
    text_np = dummy_text_tokens.cpu().numpy().astype(np.int64)
    image_embs_np = dummy_image_embs.cpu().numpy().astype(np.float32)
    token_embs_np = dummy_token_embs.cpu().numpy().astype(np.float32)

    print("\n=== TensorRT Benchmarks ===")

    # TensorRT image encoder
    image_trt = BaseEngine(args.trt_image_path, CoCa=True)
    print(f"TensorRT image_encoder: {measure_trt_time(image_trt, [image_np]):.2f} ms per run")

    # TensorRT text encoder
    text_trt = BaseEngine(args.trt_text_encoder_path, CoCa=True)
    print(f"TensorRT text_encoder: {measure_trt_time(text_trt, [text_np]):.2f} ms per run")

    # TensorRT text decoder
    decoder_trt = BaseEngine(args.trt_text_decoder_path, CoCa=True)
    print(f"TensorRT text_decoder: {measure_trt_time(decoder_trt, [image_embs_np, token_embs_np]):.2f} ms per run")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trt_image_path', type=str, default="coca_image_encoder.trt")
    parser.add_argument('--trt_text_encoder_path', type=str, default="coca_text_encoder.trt")
    parser.add_argument('--trt_text_decoder_path', type=str, default="coca_text_decoder.trt")
    args = parser.parse_args()
    main(args)
