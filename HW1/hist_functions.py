import os
import numpy as np
from numba import int32, cuda, njit
import timeit


def hist_cpu(A):
    """
     Returns
     -------
     np.array
         histogram of A of size 256
     """
    result = np.zeros(256)
    for x in A:
        result[x] = result[x] + 1
    return result


@njit
def hist_numba(A):
    """
     Returns
     -------
     np.array
         histogram of A of size 256
     """
    result = np.zeros(256)
    for x in A:
        result[x] = result[x] + 1
    return result

def hist_gpu(A):
    # Allocate the output np.array histogram C in GPU memory using cuda.to_device
    #
    # invoke the hist kernel with 1000 threadBlocks with 1024 threads each
    #
    # copy the output histogram C from GPU to cpu using copy_to_host()
    result = np.zeros(256)
    d_result = cuda.to_device(result)
    hist_kernel[1000, 1024](A, d_result)
    result = d_result.copy_to_host()
    return result

@cuda.jit
def hist_kernel(A, C):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    cuda.atomic.add(C, A[bx * cuda.blockDim.x + tx], 1)

#this is the comparison function - keep it as it is, don't change A.
def hist_comparison():
    A = np.random.randint(0,256,1000*1024)
    def timer(f):
        return min(timeit.Timer(lambda: f(A)).repeat(3, 20))
    cpu_time = timer(hist_cpu)
    numba_time = timer(hist_numba)
    cuda_time = timer(hist_gpu)
    print('CPU:', cpu_time)
    print('Numba:', numba_time)
    print('CUDA:', cuda_time)
    print('CPU speedup:', cpu_time / cuda_time)
    print('Numba speedup:', numba_time / cuda_time)

if __name__ == '__main__':
    os.environ['NUMBAPRO_NVVM'] = '/usr/local/cuda-9.0/nvvm/lib64/libnvvm.so'
    os.environ['NUMBAPRO_LIBDEVICE'] = '/usr/local/cuda-9.0/nvvm/libdevice/'
    hist_comparison()
