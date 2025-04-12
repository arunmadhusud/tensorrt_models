import torch
import numpy as np
from PIL import Image
import open_clip
import argparse

import sys
import os
# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from trt_backend import BaseEngine



def generate_caption_tensorrt(
    image_path,
    image_encoder_path,
    text_encoder_path,
    text_decoder_path,
    transform,
    max_length=50
):
    """
    Generate a caption for an image using TensorRT engines with greedy search.

    Args:
        image_path (str): Path to the image file
        image_encoder_path (str): Path to the TensorRT image encoder engine
        text_encoder_path (str): Path to the TensorRT text encoder engine
        text_decoder_path (str): Path to the TensorRT text decoder engine
        transform: Image transformation pipeline
        max_length (int): Maximum length of generated caption

    Returns:
        str: Generated caption
    """
    # Load TensorRT engines
    image_encoder_trt = BaseEngine(image_encoder_path,CoCa=True)
    text_encoder_trt = BaseEngine(text_encoder_path,CoCa=True)
    text_decoder_trt = BaseEngine(text_decoder_path,CoCa=True)

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    image_np = image_tensor.cpu().numpy().astype(np.float32)

    # Encode image with TensorRT
    image_latent, image_embs = image_encoder_trt.infer([image_np])

    # Initialize with start token
    sot_token_id = 49406  # <start_of_text> token
    eot_token_id = 49407  # <end_of_text> token
    generated_tokens = [sot_token_id]
    text = np.array([[sot_token_id]], dtype=np.int64)

    # Greedy search loop
    for _ in range(max_length):

        # Encode text so far with TensorRT
        # print(f"text shape : {text.shape}")
        text_latent, tokens_embs = text_encoder_trt.infer([text])
        tokens_embs = tokens_embs[:, :text.shape[1], :]

        # Decode with text decoder TensorRT
        logits = text_decoder_trt.infer([image_embs, tokens_embs])[0]
        # print(f"logits shape : {logits.shape}")

        # Get the prediction for the last token
        last_token_logits = logits[:, text.shape[1]-1 , :]
        # print(last_token_logits)
        predicted_token_id = np.argmax(last_token_logits)

        # Stop if end of text token is predicted
        if predicted_token_id == eot_token_id:
            break

        # Add the predicted token to our generated tokens
        generated_tokens.append(int(predicted_token_id))

        # Update the text input for the next iteration
        text = np.append(text, [[predicted_token_id]], axis=1)

    return generated_tokens


def main(args):

    model, _, transform = open_clip.create_model_and_transforms(
    model_name="coca_ViT-L-14",
    pretrained="mscoco_finetuned_laion2B-s13B-b90k"
    )

    # Generate caption
    generated_tokens = generate_caption_tensorrt(
        args.image_path,
        args.image_encoder,
        args.text_encoder,
        args.text_decoder,
        transform,
        max_length=args.max_length
    )

    int_tokens = [int(token) for token in generated_tokens]
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    generated_text = tokenizer.decode(int_tokens)
    generated_text = generated_text.split("<end_of_text>")[0].replace("<start_of_text>", "").strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate image captions using TensorRT CoCa engines.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--image_encoder", type=str, required=True, help="Path to TensorRT image encoder engine")
    parser.add_argument("--text_encoder", type=str, required=True, help="Path to TensorRT text encoder engine")
    parser.add_argument("--text_decoder", type=str, required=True, help="Path to TensorRT text decoder engine")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length of generated caption")
    args = parser.parse_args()
    main(args)
