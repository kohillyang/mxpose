'''
@author: kohill
'''
import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int32,c_float

# input type for the cos_doubles function
# must be a double array, with single dimension that is contiguous
array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')

# load the library, using numpy mechanisms
libcd = npct.load_library("libheatmaps.so", "cheatmap")

# setup the return types and argument types
libcd.genGaussionHeatmap.restype = None
libcd.genGaussionHeatmap.argtypes = [c_int32,c_int32,c_float,c_float,c_int32,array_1d_double]


def genGaussionHeatmap(height,width,x,y,stride = 1):
    out_array = np.zeros(shape = (height,width)).reshape((-1,))
    libcd.genGaussionHeatmap(height,width,x,y,stride,out_array)
    out_array = out_array.reshape((height,width))
    return out_array
if __name__ == "__main__":
    # out_array = np.zeros(shape = (512,512)).reshape((-1,))
    out_array = genGaussionHeatmap(32,32,289.807275391,88.5255126953,16)
    # out_array = out_array.reshape((512,512))
    import matplotlib.pyplot as plt
    plt.imshow(out_array)
    plt.show()
