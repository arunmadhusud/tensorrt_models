import open_clip
import torch
from PIL import Image
from utils import reshape_onnx_image_encoder,optimize_onnx,verify_image_encoder_onnx,verify_image_encoder_trt
import sys
import os
import argparse
import torch.nn.functional as F


# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from trt_export import EngineBuilder



class CoCaImageEncoder(torch.nn.Module):
    def __init__(self, model, normalize=True):
        super().__init__()
        self.model = model.visual  # Vision Transformer
        self.normalize = normalize

    def forward(self, images):
        image_latent, tokens_embs = self.model(images)
        if self.normalize:
            image_latent = F.normalize(image_latent, dim=-1)
        return image_latent, tokens_embs

def export_coca_image_encoder(model, dummy_image, onnx_path, verbose=False):
    image_encoder = CoCaImageEncoder(model)
    torch.onnx.export(
        model=image_encoder,
        args=(dummy_image,),
        f=onnx_path,
        export_params=True,
        verbose=verbose,
        input_names=["image"],
        output_names=["image_latent", "tokens_embs"],
        opset_version=18,
        dynamic_axes={
            "image": {0: "batch_size"},
            "image_latent": {0: "batch_size"},
            "tokens_embs": {0: "batch_size"},
        },
    )

def main(args):   
    
    # PyTorch model inference
    model, _, transform = open_clip.create_model_and_transforms(
    model_name="coca_ViT-L-14",
    pretrained="mscoco_finetuned_laion2B-s13B-b90k"
    )

    im = Image.open("images/cat.jpeg").convert("RGB")
    im = transform(im).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        image_latent, image_embs = model._encode_image(im)

    image_latent_np = image_latent.cpu().numpy()
    image_embs_np = image_embs.cpu().numpy()
    
    # Convert to Onnx format
    image_encoder_onnx_path = "coca_image_encoder.onnx"
    export_coca_image_encoder(model, im, image_encoder_onnx_path)
    reshape_onnx_image_encoder(image_encoder_onnx_path, image_encoder_onnx_path, batch_size=-1)
    optimize_onnx(image_encoder_onnx_path, image_encoder_onnx_path)
    print(f'Exported image encoder to ONNX at {image_encoder_onnx_path}')
    verify_image_encoder_onnx(image_encoder_onnx_path, im,image_latent_np,image_embs_np)

    image_encoder_trt_path = image_encoder_onnx_path.replace('.onnx', '.trt')
    optimal_batch_size = args.opt_batch_size
    max_batch_size = args.max_batch_size
    
    # Build TensorRT image encoder engine
    builder = EngineBuilder(verbose=False)
    builder.create_network(image_encoder_onnx_path)
    builder.create_engine(image_encoder_trt_path,optimal_batch_size, max_batch_size, fp16=args.image_fp16)
    print(f'Converted image model to TensorRT engine at {image_encoder_trt_path}')

    # Verify the TensorRT image encoder
    verify_image_encoder_trt(image_encoder_trt_path, im, image_latent_np,image_embs_np)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt_batch_size', type=int, default=1, help="Optimum batch size for TensorRT engine")
    parser.add_argument('--max_batch_size', type=int, default=3, help="Maximum batch size for TensorRT engine")
    parser.add_argument('--image_fp16', action='store_true', help="Use FP16 for image encoder")

    args = parser.parse_args()
    main(args)

