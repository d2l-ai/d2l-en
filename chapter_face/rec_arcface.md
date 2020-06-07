# Additive Angular Margin Loss for Deep Face Recognition (ArcFace)
:label:`sec_facerecognition_arcface`

After the introduction of face recognition datasets and their content structure, we move forward to train deep face recognition models.
Face representation using Deep Convolutional Neural Network (DCNN) embedding is the method of choice for face recognition. 
DCNNs map the face image, into an N-dim embedding vector that has small intra-class and large inter-class distance.

```{.python .input  n=1}
%matplotlib inline
from d2l import mxnet as d2l
import mxnet as mx
from mxnet import gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()
```

## Face-ResNet

Here, we introduce a specific convolutional network design for the face recognition task. 
As we discussed before, the input size is `112x112`  which is only half of the size (`224x224`) used in the ImageNet classification task.

Below, we use a modified ResNet (called `Face-ResNet`) backbone which starts with a `conv3x3,s1` block instead of a `conv7x7,s2` block.  
This design can keep the resolution and achieve better performance for the small input size. 
After the 4th ResNet block, a fully connected layer is applied to generate the final embedding feature. 
Compared with the global average pooling,  the fully connected layer can obtain a better performance in our experiments.

```{.python .input  n=2}

class BasicBlockV3(nn.Block):

    # Helpers
    def _conv3x3(self, channels, stride, in_channels):
        return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
            use_bias=False, in_channels=in_channels)

    def _act(self):
        if self.act_type=='prelu':
            return nn.PReLU()
        else:
            return nn.Activation(self.act_type)

    def __init__(self, channels, stride, downsample=False, in_channels=0, act_type = 'relu', **kwargs):
        super(BasicBlockV3, self).__init__(**kwargs)
        self.act_type = act_type
        self.body = nn.Sequential(prefix='')
        self.body.add(nn.BatchNorm(epsilon=2e-5))
        self.body.add(self._conv3x3(channels, 1, in_channels))
        self.body.add(nn.BatchNorm(epsilon=2e-5))
        self.body.add(self._act())
        self.body.add(self._conv3x3(channels, stride, channels))
        self.body.add(nn.BatchNorm(epsilon=2e-5))
        if downsample:
            self.downsample = nn.Sequential(prefix='')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, 
                strides=stride, use_bias=False, in_channels=in_channels))
            self.downsample.add(nn.BatchNorm(epsilon=2e-5))
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        x = self.body(x)
        if self.downsample:
            residual = self.downsample(residual)
        x = x+residual
        return x

```

Then, here is the full implementation of `FaceResNet`:

```{.python .input  n=3}
class FaceResNet(nn.Block):    


    def _conv3x3(self, channels, stride, in_channels):
        return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
            use_bias=False, in_channels=in_channels)

    def _act(self):
        if self.act_type=='prelu':
            return nn.PReLU()
        else:
            return nn.Activation(self.act_type)


    def __init__(self, layers, channels, classes, use_dropout, **kwargs):

        super(FaceResNet,self).__init__(**kwargs)
        assert len(layers)==len(channels)-1
        self.act_type = 'prelu'
        block = BasicBlockV3
        #print('use_dropout:', use_dropout)
        with self.name_scope():
            self.features = nn.Sequential(prefix='')
            self.features.add(self._conv3x3(channels[0], 1, 0))
            self.features.add(nn.BatchNorm(epsilon=2e-5))
            self.features.add(self._act())
            in_channels = channels[0]
            for i, num_layer in enumerate(layers):
                stride = 2
                self.features.add(self._make_layer(block, num_layer, channels[i+1],
                                                   stride, i+1, in_channels=in_channels))
                in_channels = channels[i+1]
            self.features.add(nn.BatchNorm(epsilon=2e-5))
            if use_dropout:
                self.features.add(nn.Dropout(0.4))
            self.features.add(nn.Flatten())
            self.features.add(nn.Dense(classes))

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0):
        layer = nn.Sequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, True, in_channels=in_channels, act_type = self.act_type,
                            prefix=''))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, in_channels=channels, act_type = self.act_type, prefix=''))
        return layer

    def forward(self, x):
        x = self.features(x)
        return x


faceresnet_spec = {18: ('basic_block', [2, 2, 2, 2], [64, 64, 128, 256, 512]),
              34: ('basic_block', [3, 4, 6, 3], [64, 64, 128, 256, 512]),
              50: ('basic_block', [3, 4, 14, 3], [64, 64, 128, 256, 512]),
              100: ('basic_block', [3, 13, 30, 3], [64, 64, 128, 256, 512])}

# Constructor
def get_faceresnet(num_layers, num_classes, use_dropout=False):
    assert num_layers in faceresnet_spec, \
        "Invalid number of layers: %d. Options are %s"%(
            num_layers, str(faceresnet_spec.keys()))
    block_type, layers, channels = faceresnet_spec[num_layers]
    net = FaceResNet(layers, channels, num_classes, use_dropout)
    initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2)
    net.initialize(initializer)
    return net



```

We can create a sample `net` with 50 layers and the embedding size of `512`.

```{.python .input  n=4}
net = get_faceresnet(50, 512, use_dropout=False)
```

Given a face image as input, the forward computation of `net` will generate a `512-dim` embedding vector.

```{.python .input  n=5}
X = np.random.uniform(size=(1, 3, 112, 112))
net(X).shape
```

## Loss Design

The previous `FaceResNet` block is used to generate the `N-dim` embedding feature for each input face image. 
A good embedding model should have the following characteristics:

1. Embeddings of the same person should be close enough.
2. Embeddings of the different persons should be far away.

To achieve the discriminative feature embedding, there are two main methods:
1. Metric learning, such as contrastive loss and triplet loss.
2. Classification loss, such as Softmax, L2-Softmax, SphereFace, CosFace and ArcFace. 

In this section, we will only talk about the classification loss as they can easily achieve state-of-the-art performance on large-scale datasets.

### Softmax Loss

Softmax loss just treats face recognition as a classification problem:

```{.python .input  n=6}

class SoftmaxBlock(nn.Block):
    def __init__(self, **kwargs):
        super(ArcMarginBlock, self).__init__(**kwargs)
        self.num_classes = kwargs.get('num_classes', 0)
        assert(self.num_classes>0)
        self.emb_size = kwargs.get('emb_size', 512)
        print('in arc margin', self.num_classes, self.emb_size)
        with self.name_scope():
            self.fc7_weight = self.params.get('fc7_weight', shape=(self.num_classes, self.emb_size))
            self.fc7_weight.initialize(init=mx.init.Normal(0.01))
        
    def forward(self, x):

        fc7 = npx.fully_connected(n, self.fc7_weight.data(x.ctx), no_bias = True, num_hidden=self.num_classes, name='fc7')
        return fc7


```

### ArcFace Loss

`SphereFace`, `CosFace`,  and `ArcFace` are all margin-based softmax losses. Compared with the vanilla Softmax loss, these loss functions can simultaneously enforce intra-class compactness and inter-class discrepancy.

Ths following `ArcMarginBlock` can be used for any combination of `SphereFace`, `CosFace` and `ArcFace` by changing the parameters. 

Parameter `margin_s`, which is set as `64.0` in most of the cases, is the scale of the embedding feature.

Parameter `margin_m` means the additive angular margin, which can be set as `0.5` for training ArcFace loss.

Parameter `margin_a` means the multiplicative angular margin, which can be set as `1.35` for training SphereFace loss.

Parameter `margin_b` means the additive cosine margin, which can be set as `0.35` for training CosFace loss.

The default parameter settings of the following `ArcMarginBlock` stand for `ArcFace`, which achieves state-of-the-art performance in many benchmarks.

```{.python .input  n=7}

class ArcMarginBlock(nn.Block):
    def __init__(self, emb_size, num_classes, margin_s=64.0, margin_m=0.5, margin_a=1.0, margin_b=0.0, **kwargs):
        super(ArcMarginBlock, self).__init__(**kwargs)
        self.margin_s = margin_s
        self.margin_m = margin_m
        self.margin_a = margin_a
        self.margin_b = margin_b
        self.num_classes = num_classes
        assert(self.num_classes>0)
        self.emb_size = emb_size
        print('in arc margin', self.num_classes, self.emb_size)
        with self.name_scope():
            self.fc7_weight = self.params.get('fc7_weight', shape=(self.num_classes, self.emb_size))
            self.fc7_weight.initialize(init=mx.init.Normal(0.01))
        
    def forward(self, x, y):

        xnorm = np.linalg.norm(x, 'fro', 1, True) + 0.00001
        nx = x / xnorm

        wnorm = np.linalg.norm(self.fc7_weight.data(x.ctx), 'fro', 1, True) + 0.00001
        nw = self.fc7_weight.data(x.ctx) / wnorm

        fc7 = npx.fully_connected(nx, nw, no_bias = True, num_hidden=self.num_classes, name='fc7')

        if self.margin_a!=1.0 or self.margin_m!=0.0 or self.margin_b!=0.0:
            gt_one_hot = npx.one_hot(y, depth = self.num_classes, on_value = 1.0, off_value = 0.0)
            if self.margin_a==1.0 and self.margin_m==0.0:
                _onehot = gt_one_hot*self.margin_b
                fc7 = fc7-_onehot
            else:
                fc7_onehot = fc7 * gt_one_hot
                cos_t = fc7_onehot
                t = np.arccos(cos_t)
                if self.margin_a!=1.0:
                    t = t*self.margin_a
                if self.margin_m>0.0:
                    t = t+self.margin_m
                margin_cos = np.cos(t)
                if self.margin_b>0.0:
                    margin_cos = margin_cos - self.margin_b
                margin_fc7 = margin_cos
                margin_fc7_onehot = margin_fc7 * gt_one_hot
                diff = margin_fc7_onehot - fc7_onehot
                fc7 = fc7+diff
        fc7 = fc7*self.margin_s
        return fc7

emb_size = 512
num_classes = 100
batch_size = 1
arc = ArcMarginBlock(emb_size=emb_size, num_classes=num_classes)
X = np.random.uniform(size=(batch_size, emb_size))
Y = np.ones((batch_size,))
fc7 = arc(X, Y)
print(fc7.shape)

```

Both `SoftmaxBlock` and `ArcMarginBlock`  output the logit of each class. 
Then, the softmax cross-entropy loss will be appended for training the network.


## Summary

* We made some customised modifications on ResNet for the face recognition task.
* ArcFace loss is defined in this section for training discriminative feature embeddings.

## Exercises

1. Try to modify the default parameter settings of ArcMarginBlock to implement different loss functions. 
2. Estimate the number of parameters in the last fully connected layer?
3. Can you try a different way to efficiently implement this embedding layer when the identity number is very large?
