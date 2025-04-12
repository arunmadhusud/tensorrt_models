import torch
from PIL import Image
import open_clip
import argparse
import torch
import numpy as np
from PIL import Image
import open_clip
from utils import verify_onnx_model, verify_trt_model, optimize_onnx, reshape_onnx
import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from trt_export import EngineBuilder

# Wrapper class for image encoding
class ImageEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)

# Wrapper class for text encoding
class TextEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)

# Function to export the image encoder
def export_image_model(model, image, onnx_path, verbose=False):        
    torch.onnx.export(
        model=model,
        args=(image,),
        f=onnx_path,
        export_params=True,
        verbose=verbose,
        input_names=["image"],
        output_names=["embedding"],
        opset_version=18,
        dynamic_axes={"image": {0: "batch_size"}, "embedding": {0: "batch_size"}},
    )

# Function to export the text encoder
def export_text_model(model, text, onnx_path, verbose=False):
    torch.onnx.export(
        model=model,
        args=(text,),
        f=onnx_path,
        export_params=True,
        verbose=verbose,
        input_names=["text"],
        output_names=["embedding"],
        opset_version=18,
        dynamic_axes={"text": {0: "batch_size"}, "embedding": {0: "batch_size"}},
    )

def main(args):

    # Load the model and tokenizer
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model.eval()  # Model in evaluation mode
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    print(f'Loaded Open_CLIP model')

    # Prepare the image and text
    image1 = preprocess(Image.open("images/dog.jpg"))
    image2 = preprocess(Image.open("images/cat.jpeg"))

    # Stack the images into a batch
    image = torch.stack([image1, image2])

    text = tokenizer(["a dog","a cat"])

    # Pytorch inference
    with torch.no_grad():
        expected_image_output = model.encode_image(image).cpu().numpy()
        expected_text_output = model.encode_text(text).cpu().numpy()

    # Convert to numpy for ONNX inference
    image_np = image.numpy().astype(np.float32)
    text_np = text.numpy().astype(np.int64)

    # Export image model
    image_onnx_path = "clip_image_encoder.onnx"
    image_encoder = ImageEncoder(model)
    export_image_model(image_encoder, image, image_onnx_path)
    reshape_onnx(image_onnx_path, image_onnx_path, batch_size=-1)
    optimize_onnx(image_onnx_path, image_onnx_path)
    print(f'Exported image model to ONNX at {image_onnx_path}')
    
    # Export text model
    text_onnx_path = "clip_text_encoder.onnx"
    text_encoder = TextEncoder(model)
    export_text_model(text_encoder, text, text_onnx_path)
    reshape_onnx( text_onnx_path,  text_onnx_path, batch_size=-1)
    optimize_onnx(text_onnx_path, text_onnx_path)
    print(f'Exported text model to ONNX at {text_onnx_path}')

    # Verify onnx models
    verify_onnx_model(image_onnx_path, image_np, expected_image_output)
    verify_onnx_model(text_onnx_path, text_np, expected_text_output)

    # Convert to TensorRT
    image_trt_path = image_onnx_path.replace('.onnx', '.trt')
    text_trt_path = text_onnx_path.replace('.onnx', '.trt')
    optimal_batch_size = args.opt_batch_size
    max_batch_size = args.max_batch_size

    builder = EngineBuilder(verbose=False)
    builder.create_network(image_onnx_path)
    builder.create_engine( image_trt_path,optimal_batch_size, max_batch_size, fp16=args.image_fp16)
    print(f'Converted image model to TensorRT engine at {image_trt_path}')
    
    builder = EngineBuilder(verbose=False)
    builder.create_network(text_onnx_path)
    builder.create_engine(text_trt_path,optimal_batch_size, max_batch_size, fp16=args.text_fp16)
    print(f'Converted text model to TensorRT engine at {text_trt_path}')

    # Verify TensorRT models
    verify_trt_model(image_trt_path, image_np, expected_image_output)
    verify_trt_model(text_trt_path, text_np, expected_text_output)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt_batch_size', type=int, default=2, help="Optimum batch size for TensorRT engine")
    parser.add_argument('--max_batch_size', type=int, default=6, help="Maximum batch size for TensorRT engine")
    parser.add_argument('--image_fp16', action='store_true', help="Use FP16 for image encoder")
    parser.add_argument('--text_fp16', action='store_true', help="Use FP16 for text encoder")

    args = parser.parse_args()
    main(args)