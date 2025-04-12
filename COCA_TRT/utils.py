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
def reshape_onnx_image_encoder(in_path, out_path, batch_size=-1):
    model = onnx.load(in_path)
    # Set batch size dynamically
    for input in model.graph.input:
        d = input.type.tensor_type.shape.dim
        d[0].dim_value = batch_size

    # Set batch size dynamically
    for output in model.graph.output:
        d = output.type.tensor_type.shape.dim
        d[0].dim_value = batch_size

    # Serialize and save the updated model
    with open(out_path, 'wb') as file_handle:
        serialized = model.SerializeToString()
        file_handle.write(serialized)

# Changing the batch size to dynamic
def reshape_onnx_text_encoder(in_path, out_path, seq_length, batch_size=-1):
    model = onnx.load(in_path)
    # Set batch size dynamically
    for input in model.graph.input:
        d = input.type.tensor_type.shape.dim
        d[0].dim_value = batch_size # batch dimesnion
        d[1].dim_value = batch_size # sequence length dimension

    # The onnx model exported from open_clip repository has got a constant value ( = seq_length + 1) based on the dummy input size given during conversion. We need to set that to dynamic.
    counter = 0
    for node in model.graph.node:
        for output in node.output:
            if "Constant" in output:
                # Skip if raw_data is too small to contain int64
                if len(node.attribute[0].t.raw_data) <8:
                    continue
                # Read raw_data as int64 and check if it equals the fixed seq_length + 1
                if len(np.frombuffer(node.attribute[0].t.raw_data, dtype=np.int64))> 0 and np.frombuffer(node.attribute[0].t.raw_data, dtype=np.int64)[0] == seq_length + 1:
                    # Overwrite with -1 in int64 (as raw bytes) to make it dynamic
                    node.attribute[0].t.raw_data = b'\xff\xff\xff\xff\xff\xff\xff\xff'
                else:
                    continue
                print(f"Updating the constant value of Node : {node.name}")

        counter+=1
    counter = 0

    for output in model.graph.output:
        d = output.type.tensor_type.shape.dim
       # Set batch size dynamically
        d[0].dim_value = batch_size
        # Set sequence length dynamically
        d[1].dim_value = batch_size

    with open(out_path, 'wb') as file_handle:
        serialized = model.SerializeToString()
        file_handle.write(serialized)

def reshape_onnx_text_decoder(in_path, out_path, seq_length, batch_size=-1):
    model = onnx.load(in_path)
    # Handle inputs (image_embs and token_embs)
    for input in model.graph.input:
        d = input.type.tensor_type.shape.dim
        # Set batch size dynamically
        if batch_size != -1:
            d[0].dim_value = batch_size
        # For token_embs, Set sequence length dynamically
        if input.name == "token_embs":
            d[1].dim_value = batch_size

    # The onnx model exported from open_clip repository has got a constant value ( = seq_length) based on the dummy input size given during conversion. We need to set that to dynamic.
    counter = 0
    for node in model.graph.node:
        for output in node.output:
            if "Constant" in output:
                # Skip if raw_data is too small to contain int64
                if len(node.attribute[0].t.raw_data) <8:
                    continue
                # Read raw_data as int64 and check if it equals the fixed seq_length
                if len(np.frombuffer(node.attribute[0].t.raw_data, dtype=np.int64))> 0 and np.frombuffer(node.attribute[0].t.raw_data, dtype=np.int64)[0] == seq_length:
                    # Overwrite with -1 in int64 (as raw bytes) to make it dynamic
                    node.attribute[0].t.raw_data = b'\xff\xff\xff\xff\xff\xff\xff\xff'
                else:
                    continue
                print(f"Updating the constant value of Node : {node.name}")
        counter+=1
    counter = 0

    # Handle outputs (logits)
    for output in model.graph.output:
        d = output.type.tensor_type.shape.dim
        # Set batch size dynamically
        d[0].dim_value = batch_size
        # Set sequence length dynamically
        d[1].dim_value = batch_size

    # Serialize and save the updated model
    with open(out_path, 'wb') as file_handle:
        serialized = model.SerializeToString()
        file_handle.write(serialized)
        print(f"Model saved to {out_path}")



# Onnx network optimization
def optimize_onnx(onnx_path, optimized_onnx_path):
    passes = onnxoptimizer.get_fuse_and_elimination_passes()
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    optimized_model = onnxoptimizer.optimize(onnx_model, passes=passes, fixed_point=False)
    onnx.checker.check_model(optimized_model)
    onnx.save(optimized_model, optimized_onnx_path)

def verify_image_encoder_onnx(onnx_path, input_image,image_latent_np,image_embs_np):
    print("ONNX model verification in progress.")
    print("Comparing the pytorch output and Onnx output.")

    # Load ONNX model
    sess = ort.InferenceSession(onnx_path)

    # Get input and output names
    input_name = sess.get_inputs()[0].name
    output_names = [out.name for out in sess.get_outputs()]

    # Run ONNX inference
    output_1, output_2 = sess.run(output_names, {input_name: input_image.cpu().numpy()})

    # Verify both outputs
    try:
        np.testing.assert_allclose(output_1,image_latent_np, rtol=1e-03, atol=1e-03)
        np.testing.assert_allclose(output_2,image_embs_np, rtol=1e-03, atol=1e-03)
        print("ONNX model verification successful.")
        return True
    except AssertionError as e:
        print("Verification failed with error:")
        print(e)
        return False

def verify_image_encoder_trt(trt_path, input_image,image_latent_np,image_embs_np):
    print("TensorRT engine verification in progress.")
    print("Comparing the pytorch output and TensorRT output.")
    trt_model = BaseEngine(trt_path,CoCa=True)
    input_image_np = input_image.cpu().numpy().astype(np.float32)  # Ensure correct dtype
    output_1, output_2 = trt_model.infer([input_image_np])
    # Verify both outputs
    try:
        np.testing.assert_allclose(output_1, image_latent_np, rtol=1e-03, atol=1e-03)
        np.testing.assert_allclose(output_2, image_embs_np, rtol=1e-03, atol=1e-03)
        print("TensorRT model verification successful.")
        return True
    except AssertionError as e:
        print("Verification failed with error:")
        print(e)
        print("\n[Note] Mismatch may be due to FP16 conversion (if --fp16 flag was used).")
        return False

def verify_text_encoder_onnx(onnx_path, text_input, text_latent_np, token_embs_np):
    print("ONNX model verification in progress.")
    print("Comparing the pytorch output and Onnx output.")

    # Load ONNX model
    sess = ort.InferenceSession(onnx_path)

    # Get input and output names
    input_name = sess.get_inputs()[0].name
    output_names = [out.name for out in sess.get_outputs()]

    # Run ONNX inference
    output_1, output_2 = sess.run(output_names, {input_name: text_input.cpu().numpy()})

    # Verify both outputs
    try:
        np.testing.assert_allclose(output_1, text_latent_np, rtol=1e-03, atol=1e-03)
        np.testing.assert_allclose(output_2, token_embs_np, rtol=1e-03, atol=1e-03)
        print("ONNX model verification successful.")
        return True
    except AssertionError as e:
        print("Verification failed with error:")
        print(e)
        return False

def verify_text_encoder_trt(trt_path, text_input, text_latent_np, token_embs_np):
    print("TensorRT engine verification in progress.")
    print("Comparing the pytorch output and TensorRT output.")
    trt_model = BaseEngine(trt_path,CoCa=True)

    seq_lenth = text_input.shape[1]
    text_input_np = text_input.cpu().numpy().astype(np.int64)  # Ensure correct dtype
    output_1, output_2 = trt_model.infer([text_input_np])
    output_2 = output_2[:,:seq_lenth,:]

    # Verify both outputs
    try:
        np.testing.assert_allclose(output_1,text_latent_np, rtol=1e-03, atol=1e-03)
        np.testing.assert_allclose(output_2,token_embs_np, rtol=1e-03, atol=1e-03)
        print("TensorRT model verification successful.")
        return True
    except AssertionError as e:
        print("Verification failed with error:")
        print(e)
        print("\n[Note] Mismatch may be due to FP16 conversion (if --fp16 flag was used).")
        return False

def verify_text_decoder_onnx(onnx_path, image_embs, token_embs, logits_np):
    print("ONNX model verification in progress.")
    print("Comparing the pytorch output and Onnx output.")
    # Load ONNX model
    sess = ort.InferenceSession(onnx_path)

    # Get input and output names
    input_names = [inp.name for inp in sess.get_inputs()]
    output_names = [out.name for out in sess.get_outputs()]

    # Run ONNX inference
    output = sess.run(output_names, {input_names[0]: image_embs.detach().cpu().numpy(),
                                     input_names[1]: token_embs.detach().cpu().numpy()})[0]  # Assuming a single output

    # Verify the output
    try:
        np.testing.assert_allclose(output, logits_np, rtol=1e-03, atol=1e-03)
        print("ONNX model verification successful.")
        return True
    except AssertionError as e:
        print("Verification failed with error:")
        print(e)
        return False

def verify_text_decoder_trt(trt_path, image_embs, token_embs, logits_np):
    print("TensorRT engine verification in progress.")
    print("Comparing the pytorch output and TensorRT output.")

    # Load TensorRT engine
    trt_model = BaseEngine(trt_path,CoCa=True)

    seq_lenth = token_embs.shape[1]
    # Convert inputs to NumPy
    image_embs_np = image_embs.detach().cpu().numpy()
    token_embs_np = token_embs.detach().cpu().numpy()

    # Run TensorRT inference
    output = trt_model.infer([image_embs_np, token_embs_np])[0]

    output = output[:,:seq_lenth,:]

    # Verify the output
    try:
        np.testing.assert_allclose(output, logits_np, rtol=1e-03, atol=1e-03)
        print("TensorRT model verification successful.")
        return True
    except AssertionError as e:
        print("Verification failed with error:")
        print(e)
        print("\n[Note] Mismatch may be due to FP16 conversion (if --fp16 flag was used).")
        return False



