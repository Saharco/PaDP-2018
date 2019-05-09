from numba import cuda
from numba import njit
import numpy as np


@cuda.jit
def convolution_kernel(flipped_kernel, padded_image, result):
    col = cuda.threadIdx.x
    row = cuda.blockIdx.x
    relevant_image = padded_image[row: row + flipped_kernel.shape[0], col: col + flipped_kernel.shape[1]]
    for inner_row in range(flipped_kernel.shape[0]):
        for inner_col in range(flipped_kernel.shape[1]):
            result[row][col] += flipped_kernel[inner_row][inner_col] * relevant_image[inner_row][inner_col]


def convolution_gpu(kernel, image):
    '''Convolve using gpu
    Parameters
    ----------
    kernel : numpy array
        A small matrix
    image : numpy array
        A larger matrix of the image pixels
            
    Return
    ------
    An numpy array of same shape as image
    '''
    flipped_kernel = np.zeros(kernel.shape)
    for row in range(kernel.shape[0]):
        for col in range(kernel.shape[1]):
            flipped_kernel[row][col] = kernel[kernel.shape[0] - row - 1][kernel.shape[1] - col - 1]

    result = np.zeros(image.shape, dtype=int)
    device_result = cuda.to_device(result)
    device_flipped_kernel = cuda.to_device(flipped_kernel)
    padded_image = np.zeros((image.shape[0] + kernel.shape[0] - 1, image.shape[1] + kernel.shape[1] - 1))
    padded_image[kernel.shape[0] // 2:-kernel.shape[0] // 2 + 1, kernel.shape[1] // 2:-kernel.shape[1] // 2 + 1] = np.copy(image)
    device_padded_image = cuda.to_device(padded_image)
    convolution_kernel[image.shape[0], image.shape[1]](device_flipped_kernel, device_padded_image, device_result)
    result = device_result.copy_to_host()
    return result



@njit
def convolution_numba(kernel, image):
    '''Convolve using numba
    Parameters
    ----------
    kernel : numpy array
        A small matrix
    image : numpy array
        A larger matrix of the image pixels
            
    Return
    ------
    An numpy array of same shape as image
    '''
    result = np.copy(image)
    flipped_kernel = np.zeros(kernel.shape)

    for row in range(kernel.shape[0]):
        for col in range(kernel.shape[1]):
            flipped_kernel[row][col] = kernel[kernel.shape[0] - row - 1][kernel.shape[1] - col - 1]

    padded_image = np.zeros((image.shape[0] + kernel.shape[0] - 1, image.shape[1] + kernel.shape[1] - 1))
    padded_image[kernel.shape[0]//2:-kernel.shape[0]//2 + 1, kernel.shape[1]//2:-kernel.shape[1]//2 + 1] = np.copy(image)
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            relevant_image = padded_image[row: row + kernel.shape[0], col: col + kernel.shape[1]]
            result[row][col] = 0
            for inner_row in range(kernel.shape[0]):
                for inner_col in range(kernel.shape[1]):
                    result[row][col] += flipped_kernel[inner_row][inner_col] * relevant_image[inner_row][inner_col]
    return result
