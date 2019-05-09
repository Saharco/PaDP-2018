import numpy  as  np
import matplotlib.pyplot as plt
import pickle
import timeit
from filters import *
from scipy.signal import convolve2d


#7X7
edge_kernel = np.array([[-3/18, -2/13, -1/10, 0, 1/10, 2/13, 3/18],
               [-3/13, -2/8, -1/5, 0, 1/5, 2/8, 3/13],
               [-3/10, -2/5, -1/2, 0, 1/2, 2/5, 3/10],
               [-3/9, -2/4, -1/1, 0, 1/1, 2/4, 3/9],
               [-3/10, -2/5, -1/2, 0, 1/2, 2/5, 3/10],
               [-3/13, -2/8, -1/5, 0, 1/5, 2/8, 3/13],
               [-3/18, -2/13, -1/10, 0, 1/10, 2/13, 3/18]])

#5X5
blur_kernel = np.array([[1/256, 4/256, 6/256, 4/256, 1/256],
               [4/256, 16/256, 24/256, 16/256, 4/256],
               [6/256, 24/256, 36/256, 24/256, 6/256],
               [4/256, 16/256, 24/256, 16/256, 4/256],
               [1/256, 4/256, 6/256, 4/256, 1/256]])

#3X3
shapen_kernel = np.array([[0, -1, 0],
                 [-1, 5, -1],
                 [0, -1, 0]])


def get_image(): 
    fname = 'data/lena.dat'
    f = open(fname,'rb')
    lena = np.array(pickle.load(f))
    f.close()
    return np.array(lena[175:390, 175:390])


# Note: Use this on your local computer to better understand what the convolution does.
def show_image(image):
    """ Plot an image with matplot

    Parameters
    ----------
    image: list
        2d list of pixels
    """
    plt.imshow(image, cmap='gray')
    plt.show()
    
def conv_comparison():
    """ Compare convolution functions run time.
    """
    image = get_image()

    def timer(kernel, f):
        return min(timeit.Timer(lambda: f(kernel, image)).repeat(10, 1))

    print("---------------------------------------------")
    print("Warning: CPU calculation will take a while...")
    print("---------------------------------------------")
    print('CPU 3X3 kernel:', timer(shapen_kernel, convolve2d))
    print('Numba 3X3 kernel:', timer(shapen_kernel, convolution_numba))
    print('CUDA 3X3 kernel:', timer(shapen_kernel, convolution_gpu))
    print("---------------------------------------------")

    print('CPU 5X5 kernel:', timer(blur_kernel, convolve2d))
    print('Numba 5X5 kernel:', timer(blur_kernel, convolution_numba))
    print('CUDA 5X5 kernel:', timer(blur_kernel, convolution_gpu))
    print("---------------------------------------------")

    print('CPU 7X7 kernel:', timer(edge_kernel, convolve2d))
    print('Numba 7X7 kernel:', timer(edge_kernel, convolution_numba))
    print('CUDA 7X7 kernel:', timer(edge_kernel, convolution_gpu))
    print("---------------------------------------------")


if __name__ == '__main__':
    os.environ['NUMBAPRO_NVVM'] = '/usr/local/cuda-9.0/nvvm/lib64/libnvvm.so'
    os.environ['NUMBAPRO_LIBDEVICE'] = '/usr/local/cuda-9.0/nvvm/libdevice/'
    conv_comparison()
