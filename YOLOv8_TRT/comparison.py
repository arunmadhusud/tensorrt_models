import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
from utils import measure_trt,measure_pytorch
import numpy as np
import argparse

def main(args):

    # Load the YOLO model
    model = YOLO('yolov8n.pt')

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    yolo_model = model.model
    yolo_model.to(device).eval()

    dummy_image = torch.randn(1, 3, 640, 640).to(device)
   
    # Pytorch Inference (excluding the NMS operation)
    pytorch_avg_time = measure_pytorch(yolo_model, dummy_image)
    print(f"PyTorch average inference time: {pytorch_avg_time:.2f} ms")

    # Convert dummy_image to numpy for TRT
    image_np = dummy_image.cpu().numpy().astype(np.float32)
    # TensorRT inference (including NMS operation using plugin: EfficientNMS_TRT)
    trt_avg_time = measure_trt(args.trt_path, image_np)

    print(f"TensorRT average inference time: {trt_avg_time:.2f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trt_path', type=str, default="yolov8x.trt")
    args = parser.parse_args()
    main(args)    




