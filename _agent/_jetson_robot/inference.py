import pycuda.driver as cuda
import pycuda.autoinit

def infer(engine, input_data):
    context = engine.create_execution_context()
    
    # 分配输入输出内存
    input_size = input_data.nbytes
    output_size = ...  # 根据模型输出维度
    d_input = cuda.mem_alloc(input_size)
    d_output = cuda.mem_alloc(output_size)
    stream = cuda.Stream()
    
    # 复制输入到设备
    cuda.memcpy_htod_async(d_input, input_data, stream)
    
    # 执行推理
    context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    
    # 从设备复制输出
    output_data = np.empty(..., dtype=np.float32)  # 根据输出尺寸设置
    cuda.memcpy_dtoh_async(output_data, d_output, stream)
    stream.synchronize()
    
    return output_data
