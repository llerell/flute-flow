import os
os.environ['PYOPENCL_CTX'] = '0'
import pyopencl as cl
import numpy as np

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# 1. Use the flag ALLOC_HOST_PTR
mf = cl.mem_flags
# Instead of copying a pre-existing array, we tell OpenCL to allocate the memory
a_g = cl.Buffer(ctx, mf.READ_WRITE | mf.ALLOC_HOST_PTR, size=100 * 4) # 4 bytes per float32

# 2. "Map" the buffer to a NumPy array
# This creates a Python view of the ACTUAL memory the GPU will use
a_np, _ = cl.enqueue_map_buffer(queue, a_g, cl.map_flags.WRITE, 0, (100,), np.float32)

# 3. Fill the array normally
a_np[:] = np.random.rand(100).astype(np.float32)

# 4. You MUST unmap before the GPU kernel runs
a_np.base.release(queue) 

# The OpenCL C Kernel
prg = cl.Program(ctx, """
__kernel void sum(__global const float *a_g, __global const float *b_g, __global float *res_g) {
  int i = get_global_id(0);
  res_g[i] = a_g[i] + b_g[i];
}
""").build()

# Execute and retrieve results
prg.sum(queue, a_np.shape, None, a_g, b_g, res_g)
res_np = np.empty_like(a_np)
cl.enqueue_copy(queue, res_np, res_g)

# Verify
print("GPU Result matches CPU:", np.allclose(res_np, a_np + b_np))