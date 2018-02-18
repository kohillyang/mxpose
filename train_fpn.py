'''
Created on Feb 18, 2018

@author: kohill
'''
from __future__ import print_function
import os

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
from tensorboardX import SummaryWriter
from data_iter_fpn import getDataLoader
from rcnn.symbol.symbol_mask_fpn import get_symbol
import mxnet as mx
import logging, os
import numpy as np

BATCH_SIZE = 8
NUM_LINKS = 19
NUM_PARTS = 19

SAVE_PREFIX = "models/resnet-101"
PRETRAINED_PREFIX = "pre/deeplab_cityscapes"
LOGGING_DIR = "logs"


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
    data_iter = getDataLoader(batch_size=BATCH_SIZE)
    for d in data_iter:
        for x in d:
            print(x.shape)
        break
    input_shape = (368, 368)
    stride = (8, 8)
    sym = get_symbol(is_train=True, numberofparts=NUM_PARTS, numberoflinks=NUM_LINKS)
    batch_size = BATCH_SIZE
    # sym = memonger.search_plan(sym,data = (batch_size,3,368,368),
    #                            label = (BATCH_SIZE,NUM_PARTS * 2 + (NUM_LINKS * 4) ,
    #                                     input_shape[0]//stride[0],input_shape[1]//stride[1]))

    model = mx.mod.Module(symbol=sym, context=[mx.gpu(g) for g in gpus],
                          label_names=['heatmaplabel', 'partaffinityglabel', 'heatmapweight','pafmapweight'])
    model.bind(data_shapes=[('data', (batch_size, 3, 512, 512))], label_shapes=[
        ('heatmaplabel', (batch_size, 414656)),
        ('partaffinityglabel', (batch_size, 829312)),
        ('heatmapweight', (batch_size,  414656)),
        ('pafmapweight', (batch_size,  829312))])

    summary_writer = SummaryWriter(LOGGING_DIR)
    if retrain:
        args, auxes = load_checkpoint("pre/rcnn_coco", 0)
    else:
        args, auxes = load_checkpoint(SAVE_PREFIX + "final", start_n_dataset)

    model.init_params(arg_params=args, aux_params=auxes, allow_missing=retrain, allow_extra=True,
                      initializer=mx.init.Xavier(magnitude=.2))
    model.init_optimizer(optimizer='rmsprop',
                         optimizer_params=(('learning_rate', 1e-4),))
    for n_data_wheel in range(ndata):
        model.save_checkpoint(SAVE_PREFIX + "final", n_data_wheel + start_n_dataset)
        for nbatch, data_batch in enumerate(data_iter):
            # imgs_batch, heatmaps_batch, pafmaps_batch, heatmaps_weight_batch, pafmaps_weight_batch = data_batch
            data = mx.nd.array(data_batch[0])
            label = [mx.nd.array(x) for x in data_batch[1:]]
            model.forward(mx.io.DataBatch(data=[data], label=label), is_train=True)
            predi = model.get_outputs()
            model.backward()
            model.update()
            losses_len = len(predi)
            global_step = nbatch + len(data_iter) * n_data_wheel
            print("{0} {1} {2}".format(global_step, n_data_wheel, nbatch), end=" ")
            for i in range(losses_len // 2):
                loss = mx.nd.sum(predi[i]).asnumpy()[0]
                summary_writer.add_scalar("heatmap_loss_{}".format(i), loss,
                                          global_step=nbatch)
                print(loss, end=" ")
            for index, i in enumerate(range(losses_len // 2, losses_len)):
                loss = mx.nd.sum(predi[i]).asnumpy()[0]
                print(loss, end=" ")
                summary_writer.add_scalar("paf_loss_{}".format(index),
                                          loss,
                                          global_step=global_step)
            print("")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train(retrain=True, gpus=[0, 1], start_n_dataset=0)