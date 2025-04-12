import torch
import torch.nn.functional as F
import open_clip
import onnx
import random
import onnx
import onnxoptimizer

from utils import reshape_onnx_text_encoder,optimize_onnx,verify_text_encoder_onnx,verify_text_encoder_trt
import sys
import os
import argparse

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from trt_export import EngineBuilder


class CoCaTextEncoder(torch.nn.Module):
    def __init__(self, model, normalize=True):
        super().__init__()
        self.model = model.text  # Text Transformer
        self.normalize = normalize

    def forward(self, text_input):
        text_latent, token_emb = self.model(text_input)
        if self.normalize:
            text_latent = F.normalize(text_latent, dim=-1)
        return text_latent, token_emb

def export_coca_text_encoder(model, dummy_text, onnx_path, verbose=False, device="cpu"):

    text_encoder = CoCaTextEncoder(model)
    torch.onnx.export(
        model=text_encoder,
        args=(dummy_text,),
        f=onnx_path,
        export_params=True,
        verbose=verbose,
        input_names=["text_input"],
        output_names=["text_latent", "token_emb"],
        opset_version=18,
        dynamic_axes={
            "text_input": {0: "batch_size", 1: "num_tokens"},
            "text_latent": {0: "batch_size"},
            "token_emb": {0: "batch_size"},
        },
    )

def main(args):
    
    # PyTorch inference
    model, _, transform = open_clip.create_model_and_transforms(
        model_name="coca_ViT-L-14",
        pretrained="mscoco_finetuned_laion2B-s13B-b90k"
        )

    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    text_tokens = tokenizer(["a dog"])
    # We need to limit the token size to 76 (default output length is 77)
    # Reference : https://github.com/mlfoundations/open_clip/issues/840
    max_length = 76
    text= text_tokens[:, :max_length]    
    model.eval()
    with torch.no_grad():
      text_latent, tokens_embs = model._encode_text(text)

    text_latent_np = text_latent.cpu().numpy()
    token_embs_np = tokens_embs.cpu().numpy()
    
    # Convert to Onnx format
    text_encoder_onnx_path = "coca_text_encoder.onnx"
    export_coca_text_encoder(model, text,onnx_path=text_encoder_onnx_path)
    reshape_onnx_text_encoder(text_encoder_onnx_path, text_encoder_onnx_path,max_length,batch_size=-1)
    optimize_onnx(text_encoder_onnx_path, text_encoder_onnx_path)
    print(f'Exported text encoder to ONNX at {text_encoder_onnx_path}')
    verify_text_encoder_onnx(text_encoder_onnx_path, text,text_latent_np, token_embs_np)
    
    # Build TensorRT image encoder engine
    text_encoder_trt_path = text_encoder_onnx_path.replace('.onnx', '.trt')
    optimal_batch_size = args.opt_batch_size
    max_batch_size = args.max_batch_size  
    builder = EngineBuilder(verbose=False)
    builder.create_network(text_encoder_onnx_path)
    builder.create_engine(text_encoder_trt_path,optimal_batch_size, max_batch_size, fp16=args.text_fp16,cocatxtEncoder=True)
    print(f'Converted text encoder to TensorRT engine at {text_encoder_trt_path}')
    # Verify the TensorRT model
    verify_text_encoder_trt(text_encoder_trt_path, text,text_latent_np, token_embs_np)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt_batch_size', type=int, default=1, help="Optimum batch size for TensorRT engine")
    parser.add_argument('--max_batch_size', type=int, default=3, help="Maximum batch size for TensorRT engine")
    parser.add_argument('--text_fp16', action='store_true', help="Use FP16 for text encoder")

    args = parser.parse_args()
    main(args)
