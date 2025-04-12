import torch
import torch.nn.functional as F
import open_clip
import onnx
import random
import onnx
import onnxoptimizer
from PIL import Image


from utils import reshape_onnx_text_decoder,optimize_onnx,verify_text_decoder_onnx,verify_text_decoder_trt
import sys
import os
import argparse

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from trt_export import EngineBuilder


class CoCaTextDecoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.text_decoder  # Text Decoder (transformer or other architecture)

    def forward(self, image_embs, token_embs):
        # Forward pass of the text decoder with both image and token embeddings
        logits = self.model(image_embs, token_embs)
        return logits

def export_coca_text_decoder(model, dummy_image_embs, dummy_token_embs, onnx_path, verbose=False, device="cpu"):
    # Wrap the model
    text_decoder = CoCaTextDecoder(model)

    # Export the model
    torch.onnx.export(
        model=text_decoder,
        args=(dummy_image_embs, dummy_token_embs),  # Pass both image embeddings and token embeddings
        f=onnx_path,
        export_params=True,
        verbose=verbose,
        input_names=["image_embs", "token_embs"],  # Names of the inputs
        output_names=["logits"],  # Name of the output
        opset_version=18,
        dynamic_axes={
            "image_embs": {0: "batch_size"},
            "token_embs": {0: "batch_size",1: "num_tokens"},
            "logits": {0: "batch_size"},
        },
    )


def main(args):

    # PyTorch inference
    model, _, transform = open_clip.create_model_and_transforms(
        model_name="coca_ViT-L-14",
        pretrained="mscoco_finetuned_laion2B-s13B-b90k"
        )
    
    im = Image.open("images/cat.jpeg").convert("RGB")
    im = transform(im).unsqueeze(0)

    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    text_tokens = tokenizer(["a dog"])
    # We need to limit the token size to 76 (default output length is 77)
    # Reference : https://github.com/mlfoundations/open_clip/issues/840
    max_length = 76
    text= text_tokens[:, :max_length]
 
    model.eval()
    with torch.no_grad():
        image_latent, image_embs = model._encode_image(im)
        text_latent, token_embs = model._encode_text(text)  

    text_decoder_onnx_path = "coca_text_decoder.onnx"
    export_coca_text_decoder(model, image_embs, token_embs, onnx_path=text_decoder_onnx_path)
    reshape_onnx_text_decoder(text_decoder_onnx_path, text_decoder_onnx_path,max_length, batch_size=-1)
    optimize_onnx(text_decoder_onnx_path, text_decoder_onnx_path)
    
    # verify onnx model
    sot_token_id=49406
    sot_token_id = torch.tensor([sot_token_id])
    text = sot_token_id.unsqueeze(0)
    with torch.no_grad():
            # Encode Text generated so far
            text_latent, token_embs = model._encode_text(text)
            logits = model.text_decoder(image_embs, token_embs)
    
    # Convert expected output to numpy array
    logits_np = logits.cpu().numpy()
    verify_text_decoder_onnx(text_decoder_onnx_path, image_embs, token_embs,logits_np)    

    text_decoder_trt_path = text_decoder_onnx_path.replace('.onnx', '.trt')
    optimal_batch_size = args.opt_batch_size
    max_batch_size = args.max_batch_size
    
    # Build TensorRT image encoder engine
    builder = EngineBuilder(verbose=False)
    builder.create_network(text_decoder_onnx_path)
    builder.create_engine(text_decoder_trt_path,optimal_batch_size, max_batch_size, fp16=args.text_fp16,cocatxtEncoder=False,cocatxtDecoder=True)
    print(f'Converted text encoder to TensorRT engine at {text_decoder_trt_path}')
    
    # verify onnx model
    verify_text_decoder_trt(text_decoder_trt_path, image_embs, token_embs,logits_np)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt_batch_size', type=int, default=1, help="Optimum batch size for TensorRT engine")
    parser.add_argument('--max_batch_size', type=int, default=3, help="Maximum batch size for TensorRT engine")
    parser.add_argument('--text_fp16', action='store_true', help="Use FP16 for text encoder")

    args = parser.parse_args()
    main(args)