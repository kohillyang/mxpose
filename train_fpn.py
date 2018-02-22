'''
Created on Feb 18, 2018

@author: kohill
'''
from __future__ import print_function
import os

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
from tensorboardX import SummaryWriter
from data_iter_fpn import getDataLoader
from resnet_v1_101_deeplab import get_symbol
import mxnet as mx
import logging, os,time
import numpy as np

BATCH_SIZE =8
NUM_LINKS = 19
NUM_PARTS = 19

SAVE_PREFIX = "models/fpn/fpn_resnet-101"
PRETRAINED_PREFIX = "pre/deeplab_cityscapes"
LOGGING_DIR = "logs/log_fpn_{}".format(int(time.time()))
if not os.path.exists(LOGGING_DIR):
    os.mkdir(LOGGING_DIR)


def load_checkpoint(prefix, epoch):
    save_dict = mx.nd.load('%s-%04d.params' % (prefix, epoch))
    arg_params = {}
    aux_params = {}

    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return arg_params, aux_params


def train(retrain=True, ndata=16, gpus=[0, 1], start_n_dataset=0):
    # data_iter = getDataLoader(batch_size=BATCH_SIZE)
    # for d in data_iter:
    #     for x in d:
    #         print(x.shape)
    #     break
    input_shape = (368, 368)
    stride = (8, 8)
    sym = get_symbol(is_train=True, numberofparts=NUM_PARTS, numberoflinks=NUM_LINKS)
    batch_size = BATCH_SIZE
    # sym = memonger.search_plan(sym,data = (batch_size,3,368,368),
    #                            label = (BATCH_SIZE,NUM_PARTS * 2 + (NUM_LINKS * 4) ,
    #                                     input_shape[0]//stride[0],input_shape[1]//stride[1]))

    label_names = []
    from rcnn.config import config
    for stride in config.RPN_FEAT_STRIDE:
        label_names.append("heatmaplabel_stride{}".format(stride))
    for stride in config.RPN_FEAT_STRIDE:
        label_names.append("heatmapweight_stride{}".format(stride))
    model = mx.mod.Module(symbol=sym, context=[mx.gpu(g) for g in gpus],
                          label_names=label_names
                          )

    class TrainIter(mx.io.DataIter):

        def __init__(self):
            self.data_loader = getDataLoader(batch_size = BATCH_SIZE)
            self.INPUT_SIZE = 368
            self.provide_data = [('data', (batch_size, 3, 368, 368))]
            from rcnn.config import config
            self.strides = config.RPN_FEAT_STRIDE
            self.provide_label =[]
            for stride in self.strides:
                heatmap_shape = (batch_size, 19,self.INPUT_SIZE//stride,self.INPUT_SIZE//stride)
                self.provide_label += \
                    [('heatmaplabel_stride{}'.format(stride), heatmap_shape)]
            for stride in self.strides:
                heatmap_shape = (batch_size, 19, self.INPUT_SIZE // stride, self.INPUT_SIZE // stride)
                self.provide_label += \
                    [('heatmapweight_stride{}'.format(stride),heatmap_shape)]
            self.data_iter = None
        def reset(self):
            pass
        def __next__(self):
            data_batch  = next(self.data_iter)
            imgs_batch, heatmaps_batch, pafmaps_batch, heatmaps_weight_batch, pafmaps_weight_batch = data_batch
            heatmap_strides =[]
            heatmap_weight_strides = []

            for stride in self.strides:
                end = self.INPUT_SIZE ** 2 / (stride ** 2) * 19
                heatmap_ = heatmaps_batch[:,:end]
                heatmap_ = heatmap_.reshape((heatmap_.shape[0],-1, self.INPUT_SIZE / stride, self.INPUT_SIZE / stride))
                # print(heatmap_.shape)
                heatmaps_batch = heatmaps_batch[:,end:]
                heatmap_strides.append(heatmap_)

                heatmapw_ = heatmaps_weight_batch[:,:end]
                heatmapw_ = heatmapw_.reshape((heatmapw_.shape[0],-1, self.INPUT_SIZE / stride, self.INPUT_SIZE / stride))

                heatmaps_weight_batch = heatmaps_weight_batch[:,end:]
                heatmap_weight_strides.append(heatmapw_)

            batch = mx.io.DataBatch(data = [mx.nd.array(imgs_batch)],
                                    label = [mx.nd.array(x)
                                             for x in heatmap_strides + heatmap_weight_strides] )
            return batch
        def next(self):
            return self.__next__()
        def __iter__(self):
            self.data_iter = iter(self.data_loader)
            return self
        def __len__(self):
            return  len(self.data_loader)
    train_data_iter = TrainIter()
    lr_scheduler = mx.lr_scheduler.FactorScheduler(step = len(train_data_iter)*2, factor = 0.1,stop_factor_lr=1e-11)
    optimizer_params = {
                        # 'momentum': 0.9,
                        # 'wd': 0.0001,
                        'learning_rate': 1e-4,
                        'lr_scheduler': lr_scheduler,
                        'rescale_grad': (1.0 / 19/46/46),
                        # 'clip_gradient': 5
                        }

    if retrain:
        args, auxes = load_checkpoint("pre/rcnn_coco", 0)
    else:
        args, auxes = load_checkpoint(SAVE_PREFIX + "final", start_n_dataset)
    #
    # model.fit(train_data= train_data_iter,
    #           epoch_end_callback = mx.callback.do_checkpoint(SAVE_PREFIX+"final"),
    #           batch_end_callback =mx.callback.Speedometer(batch_size = batch_size,frequent=100),
    #           eval_metric=mx.metric.Loss(),
    #           optimizer='rmsprop', optimizer_params=optimizer_params,
    #           arg_params=args, aux_params=auxes, begin_epoch=start_n_dataset, num_epoch=ndata,
    #           allow_missing = True,
    #           initializer=mx.initializer.Xavier(magnitude=.1)
    #           )


    model.bind(data_shapes=train_data_iter.provide_data, label_shapes=train_data_iter.provide_label)

    summary_writer = SummaryWriter(LOGGING_DIR)
    model.init_params(arg_params=args, aux_params=auxes, allow_missing=retrain, allow_extra=True,
                      initializer=mx.init.Xavier(magnitude=.2))


    model.init_optimizer(optimizer='rmsprop',
                         optimizer_params=optimizer_params)
    import matplotlib.pyplot as plt
    plt.ion()
    fig,axes = plt.subplots(1,3)
    plt.show()

    for n_data_wheel in range(ndata):
        model.save_checkpoint(SAVE_PREFIX + "final", n_data_wheel + start_n_dataset)
        for nbatch, data_batch in enumerate(train_data_iter):
            model.forward(data_batch, is_train=True)
            predi = model.get_outputs()
            global_step = nbatch + len(train_data_iter) * n_data_wheel
            loss_heat = np.sum(predi[0].asnumpy())
            loss_paf = np.sum(predi[1].asnumpy())
            print("{0} {1} {2} {3} {4}".format(global_step, n_data_wheel, nbatch ,loss_heat,loss_paf), end=" ")
            summary_writer.add_scalar("heatmap_loss", loss_heat,global_step=nbatch)
            summary_writer.add_scalar("pafmap_loss", loss_paf,global_step=nbatch)

            if nbatch % 10 ==0:
                img = np.transpose( data_batch.data[0].asnumpy()[0],(1,2,0))
                heatmap0 = np.max( predi[-1].asnumpy()[0],axis = 0)
                heatmap1 = np.max( predi[-2].asnumpy()[0],axis = 0)

                axes[0].imshow(heatmap0)
                axes[1].imshow(heatmap1)

                axes[2].imshow(img)
                plt.pause(0.001)
            print("")
            model.backward()
            model.update()
            # model.forward(mx.io.DataBatch(data=[data], label=label), is_train=True)
            # predi = model.get_outputs()
            # losses_len = len(predi)
            # global_step = nbatch + len(data_iter) * n_data_wheel
            # print("{0} {1} {2}".format(global_step, n_data_wheel, nbatch), end=" ")
            # for i in range(losses_len // 2):
            #     loss = mx.nd.sum(predi[i]).asnumpy()[0]
            #     summary_writer.add_scalar("heatmap_loss_{}".format(i), loss,
            #                               global_step=nbatch)
            #     print(loss, end=" ")
            # for index, i in enumerate(range(losses_len // 2, losses_len)):
            #     loss = mx.nd.sum(predi[i]).asnumpy()[0]
            #     print(loss, end=" ")
            #     summary_writer.add_scalar("paf_loss_{}".format(index),
            #                               loss,
            #                               global_step=global_step)
            # print("")
            # model.backward()
            # model.update()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train(retrain=True, gpus=[0, 1], start_n_dataset=0)