import tensorrt as trt
import numpy as np
from cuda import cuda, cudart

# Explicitly managing batch dimension
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

#Checks the result of a CUDA or cuDNN call for errors.
def check_cuda_err(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError("Cuda Runtime Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))
    
# Calls a CUDA function and checks its return status using check_cuda_err.
def cuda_call(call):
    err, res = call[0], call[1:]
    check_cuda_err(err)
    if len(res) == 1:
        res = res[0]
    return res

# Wrapper for cudaMemcpy which infers copy size and does error checking
def memcpy_host_to_device(device_ptr: int, host_arr: np.ndarray):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(cudart.cudaMemcpy(device_ptr, host_arr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))


# Wrapper for cudaMemcpy which infers copy size and does error checking
def memcpy_device_to_host(host_arr: np.ndarray, device_ptr: int):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(cudart.cudaMemcpy(host_arr, device_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))

class BaseEngine(object):
    def __init__(self,engine_path,CoCa = False):
        self.logger  = trt.Logger(trt.Logger.WARNING)
        self.logger.min_severity = trt.Logger.Severity.ERROR
        runtime = trt.Runtime(self.logger)
        trt.init_libnvinfer_plugins(self.logger,'') # initialize TensorRT plugins
        with open(engine_path,"rb") as f:
            serialized_engine = f.read()
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)
        assert self.engine, f"Failed to load engine from {engine_path}"
        self.context = self.engine.create_execution_context()

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        
        # print self.engine.num_io_tensors. Is thisjust a number? or input output tensor names will be stored?
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            is_input = str(self.engine.get_tensor_mode(name)) == "TensorIOMode.INPUT"
            if shape[0] < 0 : # Dynamic batch size
              assert self.engine.num_optimization_profiles > 0
              profile_shape = self.engine.get_tensor_profile_shape(name, 0) # why not self.engine.get_tensor_profile_shape(name)
              assert len(profile_shape) == 3  # minimum, optimum and maximum
              if is_input:
                  shape = profile_shape[2] # Maximum size
                  self.batch_size = shape[0] # For dynamic batch size, assign batch size as that of maximum expected batch size ( define during profile optimization)
              else:
                  assert self.batch_size is not None
                  shape = [self.batch_size, *shape[1:]]
            size = np.dtype(trt.nptype(dtype)).itemsize


            # CoCa model has a dynamic shape for the second dimension which is not supported by TensorRT for memory allocation
            # so we set it to 77 (maximum sequence length)
            if CoCa:
              if shape[1] < 0:
                shape[1] = 77

            for s in shape:
                size *=s

            # Allocate GPU memory
            allocation = cuda_call(cudart.cudaMalloc(size))
            host_allocation = None if is_input else np.zeros(shape, np.dtype(trt.nptype(dtype)))
            binding = {
                "index": i,
                "name": name,
                "dtype": np.dtype(trt.nptype(dtype)),
                "shape": list(shape),
                "allocation": allocation,
                "host_allocation": host_allocation,
            }

            self.allocations.append(allocation)
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
            # print(
            #     "{} '{}' with shape {} and dtype {}".format(
            #         "Input" if is_input else "Output", binding["name"], binding["shape"], binding["dtype"]
            #     )
            # )
        
    def infer(self, inputs:list):
        """
        Run inference on multiple input tensors.
        """
        assert len(inputs) == len(self.inputs), "Mismatch between provided inputs and expected model inputs"

        
        # Set input shapes, binding["shape"] represent the maximum shape (for dynamic shapes).
        # At inference time, we need to tell TensorRT what the actual input shape is for this run (especially for dynamic inputs).
        for i, input_tensor in enumerate(inputs):
            self.context.set_input_shape(self.inputs[i]["name"], input_tensor.shape)

        # Copy inputs to device
        for i, input_tensor in enumerate(inputs):
            memcpy_host_to_device(self.inputs[i]["allocation"], input_tensor)

        # Execute inference
        self.context.execute_v2(self.allocations)

        # Copy outputs back to host
        for o in range(len(self.outputs)):
            memcpy_device_to_host(self.outputs[o]["host_allocation"], self.outputs[o]["allocation"])

        # Prepare output arrays, Get batch size from first input
        batch_size = inputs[0].shape[0]
        out = [o["host_allocation"][:batch_size] for o in self.outputs]
        return out

            

            
            








