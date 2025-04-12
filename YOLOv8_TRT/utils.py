import onnxruntime as ort
import torch
import numpy as np
import onnx
import onnxoptimizer
import cv2
import matplotlib.pyplot as plt



import sys
import os
# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from trt_backend import BaseEngine



# Changing the batch size to dynamic
def reshape_onnx(in_path, out_path, BATCH_SIZE=-1,topk_all = 100):   
    model = onnx.load(in_path)
    # Set batch size dynamically
    for input in model.graph.input:
        d = input.type.tensor_type.shape.dim
        d[0].dim_value = BATCH_SIZE

    # Set dynamic shapes for model outputs
    # Output tensors are:
    # - num_dets: [batch_size, 1]
    #     -> Number of valid detections for each image in the batch
    # - det_boxes: [batch_size, topk_all, 4]
    #     -> Bounding box coordinates [x1, y1, x2, y2] for up to `topk_all` detections per image
    # - det_scores: [batch_size, topk_all]
    #     -> Confidence scores corresponding to each predicted box
    # - det_classes: [batch_size, topk_all]
    #     -> Predicted class indices for each box
    shapes = [BATCH_SIZE, 1, BATCH_SIZE, topk_all, 4,
            BATCH_SIZE, topk_all, BATCH_SIZE, topk_all]

    for i in model.graph.output:
        for j in i.type.tensor_type.shape.dim:
            j.dim_param = str(shapes.pop(0))

    # Serialize and save the updated model
    with open(out_path, 'wb') as file_handle:
        serialized = model.SerializeToString()
        file_handle.write(serialized)

# Onnx network optimization
def optimize_onnx(onnx_path, optimized_onnx_path):
    passes = onnxoptimizer.get_fuse_and_elimination_passes()
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    optimized_model = onnxoptimizer.optimize(onnx_model, passes=passes, fixed_point=False)
    onnx.checker.check_model(optimized_model)
    onnx.save(optimized_model, optimized_onnx_path)

def  letterbox(im,new_shape = (640, 640),color = (114, 114, 114),swap=(2, 0, 1)):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
    # Compute padding [width, height]
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im,top,bottom,left,right,cv2.BORDER_CONSTANT,value=color)  # add border
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.transpose(swap)
    im = np.ascontiguousarray(im, dtype=np.float32) / 255.
    return im, r, (dw, dh)

def rainbow_fill(size=50):  # simpler way to generate rainbow color
    cmap = plt.get_cmap('jet')
    color_list = []

    for n in range(size):
        color = cmap(n/size)
        color_list.append(color[:3])  # might need rounding? (round(x, 3) for x in color)[:3]

    return np.array(color_list)

def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    _COLORS = rainbow_fill(80).astype(np.float32).reshape(-1, 3)
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(img,(x0, y0 + 1),(x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),txt_bk_color,-1)
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img

def get_fps(trt_path):
    import time
    trt_model = BaseEngine(trt_path)
    img = np.ones((1,3,640,640))
    img = np.ascontiguousarray(img, dtype=np.float32)
    for _ in range(5):  # warmup
        _ = trt_model.infer([img])

    t0 = time.perf_counter()
    for _ in range(100):  # calculate average time
        _ = trt_model.infer([img])
    print(100/(time.perf_counter() - t0), 'FPS')


def inference(trt_path,img_path,class_names,conf=0.5):
    trt_model = BaseEngine(trt_path)
    imgsz = (640,640)
    origin_img = cv2.imread(img_path)
    img, ratio, dwdh = letterbox(origin_img,imgsz)
    fps = 0
    data = trt_model.infer([img])
    num, final_boxes, final_scores, final_cls_inds  = data
    # final_boxes, final_scores, final_cls_inds  = data
    dwdh = np.asarray(dwdh * 2, dtype=np.float32)
    final_boxes -= dwdh
    final_boxes = np.reshape(final_boxes/ratio, (-1, 4))
    final_scores = np.reshape(final_scores, (-1, 1))
    final_cls_inds = np.reshape(final_cls_inds, (-1, 1))
    dets = np.concatenate([np.array(final_boxes)[:int(num[0])], np.array(final_scores)[:int(num[0])], np.array(final_cls_inds)[:int(num[0])]], axis=-1)

    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:,
                                                            :4], dets[:, 4], dets[:, 5]
        origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                            conf=conf, class_names=class_names)
    return origin_img