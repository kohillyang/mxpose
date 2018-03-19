#encoding=utf-8
'''
Created on Mar 9, 2018

@author: kohill
'''

import numpy as np
import numpy.ctypeslib as npct
import os,cv2
from ctypes import c_int32,c_float
cu_file_path =  os.path.dirname(os.path.realpath(__file__))
array_2d_float = npct.ndpointer(dtype=np.float32, ndim=2, flags='CONTIGUOUS')
array_1d_float = npct.ndpointer(dtype=np.float32, ndim=1, flags='CONTIGUOUS')

libcd = npct.load_library("libheatpaf.so", cu_file_path)

# setup the return types and argument types
libcd.gen_pafmap.restype = None
libcd.gen_pafmap.argtypes = [array_1d_float,
                             c_int32,c_int32,
                            array_2d_float,array_2d_float,#
                            array_2d_float,
                            c_int32,
                            c_float]
def gen_pafmap(keypoints,height,width,stride):
    output1 = np.zeros(shape  =(height,width),dtype = np.float32)
    output2 = np.zeros(shape  =(height,width),dtype = np.float32)
    count = np.zeros(shape  =(height,width),dtype = np.float32)
    buffer_size = output1.size
    keypoints = np.array(keypoints).astype(np.float32)
#     print(keypoints)
    libcd.gen_pafmap(keypoints,height,width,output1,output2,count,buffer_size,stride)
    return output1,output2
def genPafs(height,width,limbs,nlimb = 19,stride = 8):
    pafs = [np.zeros(dtype = np.float32,shape = (46,46)) for _ in range(nlimb * 2)]
#     counts = [np.zeros(shape = np.float32,shape = (height,width)) for _ in range(nlimb)]
#     print(limbs)
    for limb_id,x0,y0,x1,y1 in limbs:
        p0,p1 = gen_pafmap([x0,y0,x1,y1], 46, 46, 8)
        pafs[2*limb_id] += p0
        pafs[2*limb_id + 1] += p1
    return pafs
if __name__ == "__main__":
    # out_array = np.zeros(shape = (512,512)).reshape((-1,))
#     out_array = genGaussionHeatmap(32,32,289.807275391,88.5255126953,16)
    # out_array = out_array.reshape((512,512))
    pafs  = genPafs(46,46,[[0,282.89557800292965, 107.95975952148437, 306.40877532958984, 154.21342773437499]])
    p0 = pafs [0];
    p1 = pafs [1]
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,squeeze=True)
    axes[0].imshow(cv2.resize(p0,(46,46))) 
    axes[1].imshow(p1) 

    plt.show()

if __name__ == '__main__':
    pass