#!/usr/bin/env python

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""cifar10_dist.py contains code that trains a ResNet18 network using distributed training"""

from __future__ import print_function

import collections
import d2l
import math
import os
import pandas as pd
import random
import shutil
import sys
import time
import mxnet as mx
from mxnet import autograd, gluon, kv, init, np, npx
from mxnet.gluon.model_zoo import vision
from mxnet.gluon import nn

npx.set_np()




#########################################################################################
## Only do it once, rather than on multiple workers
#########################################################################################

# demo = True

# d2l.DATA_HUB['cifar10_tiny'] = (d2l.DATA_URL + 'kaggle_cifar10_tiny.zip',
#                                 '2068874e4b9a9f0fb07ebe0ad2b29754449ccacd')

# if demo:
#     data_dir = d2l.download_extract('cifar10_tiny')
#     train_data_size = 800
#     batch_size = 1
# else:
#     data_dir = 'data/'
#     train_data_size = 50000
#     batch_size_per_gpu = 128  # 64 images in a batch
#     gpus_per_machine = 2
#     batch_size = batch_size_per_gpu * gpus_per_machine


# def reorg_cifar10_data(data_dir, valid_ratio):
#     labels = d2l.read_csv_labels(data_dir + 'trainLabels.csv')
#     d2l.reorg_train_valid(data_dir, labels, valid_ratio)
#     d2l.reorg_test(data_dir)

# reorg_cifar10_data(data_dir, valid_ratio = 0)

#########################################################################################



# Create a distributed key-value store
store = kv.create('dist')
print(store.num_workers, store.rank)

class SplitSampler(gluon.data.sampler.Sampler):
    """ Split the dataset into `num_parts` parts and sample from the 
        part with index `part_index`

    Parameters
    ----------
    length: int
      Number of examples in the dataset
    num_parts: int
      Partition the data into multiple parts
    part_index: int
      The index of the part to read from
    """
    def __init__(self, length, num_parts=1, part_index=0):
        # Compute the length of each partition
        self.part_len = length // num_parts
        # Compute the start index for this partition
        self.start = self.part_len * part_index
        # Compute the end index for this partition
        self.end = self.start + self.part_len

    def __iter__(self):
        # Extract examples between `start` and `end`, shuffle and return them.
        indices = list(range(self.start, self.end))
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.part_len

    
class SplitBatchSampler(gluon.data.sampler.Sampler):
    
    def __init__(self, length, batch_size, num_parts=1, part_index=0, last_batch='keep'):
        self.part_len = length // num_parts
        self._batch_size = batch_size
        self.start = self.part_len * part_index
        self.end = self.start + self.part_len
        self._last_batch = last_batch
        self._prev = []

    def __iter__(self):
        indices = list(range(self.start, self.end))
        random.shuffle(indices)
        batch, self._prev = self._prev, []
        for i in indices:
            batch.append(i)
            if len(batch) == self._batch_size:
                yield batch
                batch = []
        if batch:
            if self._last_batch == 'keep':
                yield batch
            elif self._last_batch == 'discard':
                return
            elif self._last_batch == 'rollover':
                self._prev = batch
            else:
                raise ValueError(
                    "last_batch must be one of 'keep', 'discard', or 'rollover', " \
                    "but got %s"%self._last_batch)

    def __len__(self):
        if self._last_batch == 'keep':
            return (len(self._sampler) + self._batch_size - 1) // self._batch_size
        if self._last_batch == 'discard':
            return len(self._sampler) // self._batch_size
        if self._last_batch == 'rollover':
            return (len(self._prev) + len(self._sampler)) // self._batch_size
        raise ValueError(
            "last_batch must be one of 'keep', 'discard', or 'rollover', " \
            "but got %s"%self._last_batch)

class Residual(nn.HybridBlock):
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1,
                               strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1,
                                   strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def hybrid_forward(self, F, X):
        Y = F.npx.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.npx.relu(Y + X)
    
def resnet18(num_classes):
    net = nn.HybridSequential()
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))

    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.HybridSequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(Residual(num_channels))
        return blk

    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net


# Load the training and test data
transform_train = gluon.data.vision.transforms.Compose([
    # Magnify the image to a square of 40 pixels in both height and width
    gluon.data.vision.transforms.Resize(40),
    # Randomly crop a square image of 40 pixels in both height and width to
    # produce a small square of 0.64 to 1 times the area of the original
    # image, and then shrink it to a square of 32 pixels in both height and
    # width
    gluon.data.vision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                                   ratio=(1.0, 1.0)),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor(),
    # Normalize each channel of the image
    gluon.data.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                           [0.2023, 0.1994, 0.2010])])

transform_test = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.ToTensor(),
    gluon.data.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                           [0.2023, 0.1994, 0.2010])])

train_ds, test_ds = [
    gluon.data.vision.ImageFolderDataset(data_dir + "train_valid_test/" + folder)
    for folder in ['train','test']]

train_iter = gluon.data.DataLoader(
    train_ds.transform_first(transform_train), 
#     batch_size, 
#     shuffle=True,
#     sampler=gluon.data.sampler.BatchSampler,
#     sampler=SplitSampler(train_data_size, store.num_workers, store.rank),
    batch_sampler=SplitBatchSampler(train_data_size, batch_size, store.num_workers, store.rank, last_batch='keep'),
#     last_batch='keep'
) 

test_iter = gluon.data.DataLoader(
    test_ds.transform_first(transform_test), batch_size, shuffle=False,
    # sampler=SplitSampler(test_data_size, store.num_workers, store.rank),
#     last_batch='keep'
) 


# train_iter = gluon.data.DataLoader(
#     gluon.data.vision.CIFAR10(train=True, transform=transform_train), batch_size, # shuffle=True,
#     sampler=SplitSampler(50000, store.num_workers, store.rank)
# )

# test_iter = gluon.data.DataLoader(
#     gluon.data.vision.CIFAR10(train=False, transform=transform_train), batch_size, shuffle=False,
#     # sampler=SplitSampler(test_data_size, store.num_workers, store.rank),
#     last_batch='keep') 



def train(net, train_iter, test_iter, num_epochs, lr, wd, ctx, lr_period,
          lr_decay):

    # Use SGD optimizer. Ask trainer to use the distributor kv store.
    trainer = gluon.Trainer(net.collect_params(), 'sgd', 
                            {'learning_rate': lr, 'momentum': 0.9, 'wd': wd}, 
                            kvstore=store)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        for X, y in train_iter:
            y = y.astype('float32').as_in_context(ctx)
            with autograd.record():
                y_hat = net(X.as_in_context(ctx))
                l = loss(y_hat, y).sum()
            l.backward()
            trainer.step(batch_size)
            train_l_sum += float(l)
            train_acc_sum += float((y_hat.argmax(axis=1) == y).sum())
            n += y.size
        time_s = "time %.2f sec" % (time.time() - start)
#         if valid_iter is not None:
#             valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
#             epoch_s = ("epoch %d, loss %f, train acc %f, valid acc %f, "
#                        % (epoch + 1, train_l_sum / n, train_acc_sum / n,
#                           valid_acc))
#         else:
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        epoch_s = ("epoch %d, loss %f, train acc %f, test acc %f" %
                   (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
        print(epoch_s + time_s + ', lr ' + str(trainer.learning_rate))
        sys.stdout.flush()
        
    


# Create the context (a list of all GPUs to be used for training)
# ctx = [mx.gpu(i) for i in range(gpus_per_machine)]
ctx = d2l.try_gpu()
num_epochs, lr, wd = 1, 0.1, 5e-4
lr_period, lr_decay  = 80, 0.1
num_classes = 10
# net = get_net(ctx)

net = resnet18(num_classes)
net.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
net.hybridize()
train(net, train_iter, test_iter, num_epochs, lr, wd, ctx, lr_period,
      lr_decay)
