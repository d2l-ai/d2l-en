# Model Parallelism of Training ArcFace
:label:`sec_facerecognition_paralltraining`\

```{.python .input  n=2}
%matplotlib inline
from d2l import mxnet as d2l
import mxnet as mx
from mxnet import gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()
```

As we can see in the previous section, training ArcFace is quite easy. 
However, the size of the linear transformation matrix increases linearly with the identity number. 
Each GPU card needs to hold a large matrix of `NxD`, where `N` is the number of classes and `D` is the embedding size. 
In large-scale face recognition problem, `N` can be very large(e.g. N>1M). In such a case, OOM appears.

## Model Parallelism

The solution is to adopt model parallelism to split the large classification matrix into different GPU cards. 
For example, if we have `N=1M` classes and 8 GPU cards, each GPU card will only need to keep a `(1M//8, D)` matrix.

## Implementation

As we split the whole classification matrix into several sub-matrixs across the GPUs, 
we need to implement some forward/backward methods by ourselves instead of calling the off-the-shelf functions.

```{.python .input  n=3}
dataset = "faces-casia-ultrasmall"
data_dir = d2l.download_extract(dataset, dataset)
batch_size = 512
image_size, num_classes = d2l.read_facerec_meta(data_dir)
loader, val_set = d2l.load_data_face_rec(data_dir, ['lfw_small'], batch_size)
print('image_size:', image_size)
print('num_classes:', num_classes)
assert 'lfw_small' in val_set

```

Initialize all GPU contexts, determine the number of classes per device, and calculate the class offset of each device:

```{.python .input  n=5}
ctx = d2l.try_all_gpus()
num_ctx = len(ctx)
if num_classes % num_ctx==0:
    ctx_num_classes = num_classes//num_ctx
else:
    ctx_num_classes = num_classes//num_ctx+1

ctx_class_start = []
for i in range(num_ctx):
    _c = i*ctx_num_classes
    ctx_class_start.append(_c)

print('num_ctx:', num_ctx)
print('ctx num classes:', ctx_num_classes)
print('class offsets:', ctx_class_start)
```

Define the feature extraction block and initialize a network of 18 layers:

```{.python .input  n=6}
class FeatBlock(nn.Block):
    def __init__(self, num_layers, emb_size, use_dropout, is_train, **kwargs):
        super(FeatBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.feat_net = nn.Sequential(prefix='')
            self.feat_net.add(d2l.get_faceresnet(num_layers, emb_size, use_dropout, do_init=is_train))
            self.is_train = is_train

    def forward(self, x):
        feat = self.feat_net(x)
        if self.is_train:
            mean = np.mean(feat, axis=[0])
            var = np.var(feat, axis=[0])
            var = np.sqrt(var + 2e-5)
            feat = (feat - mean) / var
        return feat

use_dropout = False
if num_classes<=20000:
    use_dropout = True
    
emb_size = 256
    
net = FeatBlock(18, emb_size, use_dropout, is_train=True)
net.collect_params().reset_ctx(ctx)
```

To split the whole classification matrix into smaller ones over all GPU devices, we define one `ArcMarginBlock` per device:

```{.python .input  n=8}

cls_nets = []
for i in range(num_ctx):
    cls_net = d2l.ArcMarginBlock(emb_size, ctx_num_classes)
    #cls_net.initialize(init=mx.init.Normal(0.01))
    cls_net.collect_params().reset_ctx(ctx[i])
    cls_nets.append(cls_net)
```

Use a MxNet NumPy array cache container:

```{.python .input  n=9}
class NPCache():
    def __init__(self):
        self._cache = {}

    def get(self, context, name, shape):
        key = "%s_%s"%(name, context)
        #print(key)
        if not key in self._cache:
            v = mx.np.zeros( shape=shape, ctx = context)
            self._cache[key] = v
        else:
            v = self._cache[key]
        return v

    def get2(self, context, name, arr):
        key = "%s_%s"%(name, context)
        
        if not key in self._cache:
            v = mx.np.zeros( shape=arr.shape, ctx = context)
            self._cache[key] = v
        else:
            v = self._cache[key]
            arr.copyto(v)
        return v

cache = NPCache()
```

Now we can start training the model and split the whole process into the following steps:
1. Forward input images on all devices and aggregate all embedding features on CPU.
2. Use the above aggregated embedding as input and forward it on all defined ArcMarginBlocks. Each device generates parts of class logits.
3. Aggregate partial logits from all devices into complete logits.
4. Calculate the gradient of logits and calculate back-propagation manually.
5. Calculate the gradients of the embedding tensor.

We show sample codes below. For the detailed parallel training process, please check the d2l python package.

```{.python .input  n=10}
tmp_ctx = ctx[0]
cpu_ctx = mx.cpu()
def forward_emb(data, label):

    fc1_out_list = []
    fc1_list_cpu = []
    with autograd.record():
        for _data, _label in zip(data, label):
            fc1 = feat_net(_data)
            fc1_out_list.append(fc1)
    for _fc1 in fc1_out_list:
        fc1_cpu = _fc1.as_in_ctx(cpu_ctx)
        fc1_list_cpu.append(fc1_cpu)
    global_fc1 = cache.get(cpu_ctx, 'global_fc1_cpu', (batch_size, emb_size))
    mx.np.concatenate(fc1_list_cpu, axis=0, out=global_fc1)
    return global_fc1, fc1_out_list
```

To get partial logits from each device:

```{.python .input  n=11}
def forward_logits(global_fc7, y):
    _xlist = []
    _ylist = []
    for i, cls_net in enumerate(cls_nets):
        _ctx = ctx[i]
        _y = cache.get2(_ctx, 'ctxy', y)
        _y -= ctx_class_start[i]
        _x = cache.get2(_ctx, 'ctxfc1', global_fc1)
        _xlist.append(_x)
        _ylist.append(_y)
    fc1_list = []
    fc7_list = []
    with autograd.record():
        for i, cls_net in enumerate(cls_nets):
            _ctx = ctx[i]
            _x = _xlist[i]
            _y = _ylist[i]
            _x.attach_grad()
            _fc7 = cls_net(_x, _y)
            fc7_list.append(_fc7)
            fc1_list.append(_x)
    return fc1_list, fc7_list
```

Start training the model. (Some detail parall training code was omitted here, please check d2l python package). For a large number of classes, we can significantly save GPU memory and the training time.

```{.python .input  n=12}
num_epochs = 5

d2l.train_ch_facerec_parall(net, cls_nets, loader, num_epochs, ctx)
```

## Face Verification

Face verification accuracy on `lfw_small`:

```{.python .input  n=14}
test_net = FeatBlock(18, emb_size, use_dropout, is_train=False, params=net.collect_params())
lfw_xnorm, lfw_acc, lfw_thresh = d2l.test_face_11(val_set['lfw_small'], test_net, ctx, batch_size)
print('LFW-Small Accuracy:', lfw_acc)
```

## Summary

1. How to train ArcFace model by model parallelism.
2. Model parallelism can significantly save GPU memory and speed up the training process.
