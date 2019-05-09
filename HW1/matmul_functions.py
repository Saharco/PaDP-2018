import os
import numpy as np
from numba import njit, cuda
import timeit


def matmul_trivial(X, Y):
    result = np.zeros((X.shape[0], Y.shape[1]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[1]):
            for k in range(X.shape[1]):
                result[i][j] += X[i][k] * Y[k][j]
    return result

@njit
def matmul_numba(X, Y):
    result = np.zeros((X.shape[0], Y.shape[1]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[1]):
            for k in range(X.shape[1]):
                result[i][j] += X[i][k] * Y[k][j]
    return result


def matmul_gpu(X, Y):
    # Allocate the output matrix in GPU memory using cuda.to_device
    #
    # invoke the dot kernel with 1 threadBlock with 1024 threads
    #
    # copy the output matrix from GPU to cpu using copy_to_host()
    result = np.zeros((X.shape[0], Y.shape[1]))
    d_result = cuda.to_device(result)
    matmul_kernel[1, 1024](X, Y, d_result)
    result = d_result.copy_to_host()
    return result

@cuda.jit
def matmul_kernel(A, B, C):
    tx = cuda.threadIdx.x
    to_calculate = range(tx, C.size, 1024)
    for i in to_calculate:
        row_index = int(i / C.shape[1])
        col_index = i % C.shape[1]
        for k in range(A.shape[1]):
            C[row_index][col_index] += A[row_index][k] * B[k][col_index]

#this is the comparison function - keep it as it is, don't change X or Y.
def matmul_comparison():
    X = np.random.randn(784, 128)
    Y = np.random.randn(128, 64)
    def timer(f):
        return min(timeit.Timer(lambda: f(X, Y)).repeat(3, 100))


    #print('Python:', timer(matmul_trivial))
    print('Numpy:', timer(np.matmul))
    print('Numba:', timer(matmul_numba))
    print('CUDA:', timer(matmul_gpu))

if __name__ == '__main__':
    os.environ['NUMBAPRO_NVVM'] = '/usr/local/cuda-9.0/nvvm/lib64/libnvvm.so'
    os.environ['NUMBAPRO_LIBDEVICE'] = '/usr/local/cuda-9.0/nvvm/libdevice/'
    matmul_comparison()
