import mxnet as mx
import  os,cv2
import  numpy as np
import matplotlib.pyplot as plt
from  train_fpn import  SAVE_PREFIX,get_symbol
_,args,auxes = mx.model.load_checkpoint(SAVE_PREFIX+"final",1)
module = mx.mod.Module(symbol = get_symbol(is_train=False),data_names=['data'],label_names=[])
module.bind(data_shapes=[('data',(1,3,512,512))])
# mx.visualization.plot_network(get_symbol(is_train=True)).view()
module.init_params(arg_params = args,aux_params = auxes,allow_extra=True,allow_missing=True)
for root_dir,_,names in os.walk("/home/kohill/hszc/data/coco/val2014"):
    for name in names:
        path = os.path.join(root_dir,name)
        fig,axes = plt.subplots(1,2)
        img = cv2.imread(path)

        img = cv2.resize(img,(0,0),fx = 512.0/np.max(img.shape[:2]),fy = 512.0/np.max(img.shape[:2]))
        # img_padded = np.zeros(dtype=np.uint8,shape=(512,512,3))
        # img_padded[:img.shape[0],:img.shape[1],:] = img[:,:,:]
        img_transpose = np.transpose(img,(2,0,1))[np.newaxis]

        data_batch   = mx.io.DataBatch(data = [mx.nd.array(img_transpose)])
        module.forward(data_batch)
        heatmap= module.get_outputs()[1].asnumpy()[0]
        axes[0].imshow(np.max(heatmap,axis = 0))
        axes[1].imshow(img)
        # print(result.shape)
        plt.show()