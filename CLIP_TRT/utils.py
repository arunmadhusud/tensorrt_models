import onnxruntime as ort
import torch
import numpy as np
import onnx
import onnxoptimizer

import sys
import os
# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from trt_backend import BaseEngine

# Changing the batch size to dynamic
def reshape_onnx(in_path, out_path, batch_size=-1):
    model = onnx.load(in_path)
    d = model.graph.input[0].type.tensor_type.shape.dim
    d[0].dim_value = batch_size
    for output in model.graph.output:
        d = output.type.tensor_type.shape.dim
        d[0].dim_value = batch_size

    with open(out_path, 'wb') as file_handle:
        serialized = model.SerializeToString()
        file_handle.write(serialized)

def optimize_onnx(onnx_path, optimized_onnx_path):
    passes = onnxoptimizer.get_fuse_and_elimination_passes()
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    optimized_model = onnxoptimizer.optimize(onnx_model, passes=passes, fixed_point=False)
    onnx.checker.check_model(optimized_model)
    onnx.save(optimized_model, optimized_onnx_path)


def verify_onnx_model(onnx_path, input_data, expected_output):
    print("ONNX model verification in progress.")
    print("Comparing the pytorch output and Onnx output.")
    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    output = sess.run([output_name], {input_name: input_data})[0]
    try:
        np.testing.assert_allclose(output, expected_output, rtol=1e-03, atol=1e-03)
        print("ONNX model verification successful.")
    except AssertionError as e:
        print("Verification failed with error:")
        print(e)
    return output

def verify_trt_model(trt_path, input_data, expected_output):
    print("TensorRT engine verification in progress.")
    print("Comparing the pytorch output and TensorRT output.")
    trt_model = BaseEngine(trt_path)
    output = trt_model.infer([input_data])[0]
    try:
        np.testing.assert_allclose(output, expected_output, rtol=1e-03, atol=1e-03)
        print("TensorRT model verification successful.")
        return True
    except AssertionError as e:
        print("Verification failed with error:")
        print(e)
        print("\n[Note] Mismatch may be due to FP16 conversion (if --fp16 flag was used).")
        return False


