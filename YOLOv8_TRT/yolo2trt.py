import numpy as np
import cv2
import urllib.request
from PIL import Image
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
import torchvision
from skimage import io
from skimage.transform import resize
import torch.nn as nn
import random
import onnx
from utils import reshape_onnx, optimize_onnx
import argparse

import sys
import os
# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from trt_export import EngineBuilder



class TRT_NMS(torch.autograd.Function):
    '''TensorRT NMS operation'''
    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version="1",
        score_activation=0,
        score_threshold=0.25,
    ):
        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4)
        det_scores = torch.randn(batch_size, max_output_boxes)
        det_classes = torch.randint(0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)
        return num_det, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(g,boxes,scores,background_class=-1,box_coding=1,
                 iou_threshold=0.45,max_output_boxes=100,
                 plugin_version="1",score_activation=0,score_threshold=0.25):
        out = g.op("TRT::EfficientNMS_TRT",
                   boxes,
                   scores,
                   background_class_i=background_class,
                   box_coding_i=box_coding,
                   iou_threshold_f=iou_threshold,
                   max_output_boxes_i=max_output_boxes,
                   plugin_version_s=plugin_version,
                   score_activation_i=score_activation,
                   score_threshold_f=score_threshold,
                   outputs=4)
        nums, boxes, scores, classes = out
        return nums, boxes, scores, classes

class ORT_NMS(torch.autograd.Function):
    '''ONNX-Runtime NMS operation'''
    @staticmethod
    def forward(ctx,
                boxes,
                scores,
                max_output_boxes_per_class=torch.tensor([100]),
                iou_threshold=torch.tensor([0.45]),
                score_threshold=torch.tensor([0.25])):
        device = boxes.device
        batch = scores.shape[0]
        num_det = random.randint(0, 100)
        batches = torch.randint(0, batch, (num_det,)).sort()[0].to(device)
        idxs = torch.arange(100, 100 + num_det).to(device)
        zeros = torch.zeros((num_det,), dtype=torch.int64).to(device)
        selected_indices = torch.cat([batches[None], zeros[None], idxs[None]], 0).T.contiguous()
        selected_indices = selected_indices.to(torch.int64)
        return selected_indices

    @staticmethod
    def symbolic(g, boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold):
        return g.op("NonMaxSuppression", boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)

class ONNX_TRT(nn.Module):
    '''onnx module with TensorRT NMS operation.'''
    def __init__(self, max_obj=100, iou_thres=0.45, score_thres=0.25, max_wh=None ,device=None, n_classes=80):
        super().__init__()
        assert max_wh is None
        self.device = device if device else torch.device('cpu')
        self.background_class = -1,
        self.box_coding = 1,
        self.iou_threshold = iou_thres
        self.max_obj = max_obj
        self.plugin_version = '1'
        self.score_activation = 0
        self.score_threshold = score_thres
        self.n_classes=n_classes

    def forward(self, x):
        boxes = x[:, :, :4]
        scores = x[:, :, 4:]
        num_det, det_boxes, det_scores, det_classes = TRT_NMS.apply(boxes, scores, self.background_class, self.box_coding,
                                                                    self.iou_threshold, self.max_obj,
                                                                    self.plugin_version, self.score_activation,
                                                                    self.score_threshold)
        return num_det, det_boxes, det_scores, det_classes

class ONNX_ORT(nn.Module):
    '''onnx module with ONNX-Runtime NMS operation.'''
    def __init__(self, max_obj=100, iou_thres=0.45, score_thres=0.25, max_wh=640, device=None, n_classes=80):
        super().__init__()
        self.device = device if device else torch.device("cpu")
        self.max_obj = torch.tensor([max_obj]).to(device)
        self.iou_threshold = torch.tensor([iou_thres]).to(device)
        self.score_threshold = torch.tensor([score_thres]).to(device)
        self.max_wh = max_wh # if max_wh != 0 : non-agnostic else : agnostic
        self.convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                           dtype=torch.float32,
                                           device=self.device)
        self.n_classes=n_classes

    def forward(self, x):
        boxes = x[:, :, :4]
        scores = x[:, :, 4:]
        boxes @= self.convert_matrix
        max_score, category_id = scores.max(2, keepdim=True)
        dis = category_id.float() * self.max_wh
        nmsbox = boxes + dis
        max_score_tp = max_score.transpose(1, 2).contiguous()
        selected_indices = ORT_NMS.apply(nmsbox, max_score_tp, self.max_obj, self.iou_threshold, self.score_threshold)
        X, Y = selected_indices[:, 0], selected_indices[:, 2]
        selected_boxes = boxes[X, Y, :]
        selected_categories = category_id[X, Y, :].float()
        selected_scores = max_score[X, Y, :]
        X = X.unsqueeze(1).float()
        return torch.cat([X, selected_boxes, selected_categories, selected_scores], 1)

class End2End(nn.Module):
    '''export onnx or tensorrt model with NMS operation.'''
    def __init__(self, model, max_obj=100, iou_thres=0.45, score_thres=0.25, max_wh=None, device=None, n_classes=80):
        super().__init__()
        device = device if device else torch.device('cpu')
        assert isinstance(max_wh,(int)) or max_wh is None
        self.model = model.to(device)
        self.patch_model = ONNX_TRT if max_wh is None else ONNX_ORT
        self.end2end = self.patch_model(max_obj, iou_thres, score_thres, max_wh, device, n_classes)
        self.end2end.eval()

    def forward(self, x):
        x = self.model(x)
        x_t = x[0].transpose(1,2).contiguous()
        x = self.end2end(x_t)
        return x

# Function to export the image encoder
def export_onnx_model(model, image, onnx_path, verbose=False):        
    output_names = ['num_dets', 'det_boxes', 'det_scores', 'det_classes']
    dynamic_axes={"image": {0: "batch_size"}, "num_dets": {0: "batch_size"}, "det_boxes": {0: "batch_size"}, "det_scores": {0: "batch_size"},"det_classes": {0: "batch_size"}}

    torch.onnx.export(model,image,onnx_path, verbose=verbose, opset_version=12, input_names=['images'],
                            output_names=output_names,
                            dynamic_axes=dynamic_axes)
    

def main(args):
    # Load the YOLO model
    model = YOLO('yolov8n.pt')

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    yolo_model = model.model
    # Move the model to the GPU
    yolo_model.to(device).eval()

    onnx_conv = True
    if onnx_conv : max_wh=None
    else : max_wh = 640

    onnx_model = End2End(yolo_model,max_obj=100, iou_thres=0.45, score_thres=0.25, max_wh=max_wh, device=device, n_classes=80).to(device)
    
    dummy_input = torch.randn(1, 3, 640, 640).to(device)
    onnx_path = 'yolov8x.onnx'
    onnx_model.eval()
    export_onnx_model(onnx_model,dummy_input,onnx_path, verbose=False)
    reshape_onnx(onnx_path, onnx_path, BATCH_SIZE=-1)
    optimize_onnx(onnx_path, onnx_path)
    print(f'Exported pytorch model to ONNX at {onnx_path}')

    # Convert to TensorRT
    trt_path = onnx_path.replace('.onnx', '.trt')
    optimal_batch_size = args.opt_batch_size
    max_batch_size = args.max_batch_size

    builder = EngineBuilder(verbose=False)
    builder.create_network(onnx_path)
    builder.create_engine( trt_path,optimal_batch_size, max_batch_size, fp16=args.fp16)
    print(f'Converted pytorch model to TensorRT engine at {trt_path}')

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt_batch_size', type=int, default=1, help="Optimum batch size for TensorRT engine")
    parser.add_argument('--max_batch_size', type=int, default=3, help="Maximum batch size for TensorRT engine")
    parser.add_argument('--fp16', action='store_true', help="Use FP16 for image encoder")
    parser.add_argument('--onnx_conv', action='store_true', help="Convert model to Onnx only")

    args = parser.parse_args()
    main(args)