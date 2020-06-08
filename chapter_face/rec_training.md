# Training ArcFace
:label:`sec_facerecognition_training`\

```{.python .input  n=1}
%matplotlib inline
from d2l import mxnet as d2l
import mxnet as mx
from mxnet import gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()
```

After we are familiar with data, network and loss function of the face recognition task, we can start to train a model.

## Data Load

To conduct a fast experiment, we just load a subset of `CASIA`, which contains 0.3K identities. We also employ a small validation dataset called `lfw_small` to check the accuracy of the model.

```{.python .input  n=2}
dataset = "faces-casia-ultrasmall"
data_dir = d2l.download_extract(dataset, dataset)
batch_size = 512
image_size, num_classes = d2l.read_facerec_meta(data_dir)
loader, val_set = d2l.load_data_face_rec(data_dir, ['lfw_small'], batch_size)
print('image_size:', image_size)
print('num_classes:', num_classes)
assert 'lfw_small' in val_set
```

## Model Training

Define the ArcFace training block as below. Note that we add a `Dropout`  to avoid overfitting when the number of identities is small.

```{.python .input  n=3}
class TrainBlock(nn.Block):
    def __init__(self, num_layers, emb_size, num_classes, use_dropout, **kwargs):
        super(TrainBlock, self).__init__(**kwargs)

        with self.name_scope():
            self.feat_net = nn.Sequential(prefix='')
            self.feat_net.add(d2l.get_faceresnet(num_layers, emb_size, use_dropout, do_init=True))
            self.margin_block = d2l.ArcMarginBlock(emb_size, num_classes)
            
    def forward(self, x, y):
        feat = self.feat_net(x)

        mean = np.mean(feat, axis=[0])
        var = np.var(feat, axis=[0])
        var = np.sqrt(var + 2e-5)
        feat = (feat - mean) / var

        fc7 = self.margin_block(feat,y)
        return fc7

use_dropout = False
if num_classes<=20000:
    use_dropout = True

```

Initialize a `FaceResNet-18` network with the embedding size of 256.

```{.python .input  n=4}

net = TrainBlock(18, 256, num_classes, use_dropout)

```

Now we can start training the model. 
The loss function and accuracy calculation here are similar to those used in image classification.
For a fast experiment, we just train 5 epochs. 
We can train more epochs on a larger dataset to achieve better performance.

```{.python .input  n=5}
num_epochs, lr, wd, ctx = 5, 0.1, 5e-4, d2l.try_all_gpus()
loss = gluon.loss.SoftmaxCrossEntropyLoss()
net.collect_params().reset_ctx(ctx)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': lr, 'wd': wd})
d2l.train_ch_facerec(net, loader, loss, trainer, num_epochs, ctx)
```

## Model Testing

For model testing, we define a new block, which uses the embedding layer as output and copies the learned weights from the trained network.

```{.python .input  n=6}
class TestBlock(nn.Block):
    def __init__(self, num_layers, emb_size, use_dropout, **kwargs):
        super(TestBlock, self).__init__(**kwargs)

        with self.name_scope():
            self.feat_net = nn.Sequential(prefix='')
            self.feat_net.add(d2l.get_faceresnet(num_layers, emb_size, use_dropout, do_init=False))

    def forward(self, x):
        feat = self.feat_net(x)
        return feat

test_net = TestBlock(18, 256, use_dropout, params=net.collect_params())



```

The function to get the feature embedding:

```{.python .input  n=7}
def get_embedding(img):
    img = np.array(img, ctx=ctx[0], dtype='uint8')
    img = np.expand_dims(img.transpose(2, 0, 1), axis=0).astype('float32') / 255.0
    emb = test_net(img)
    return emb
```

After the training is finished, we can easily test the face verification accuracy on some public test sets. 
For example, `LFW` (Labeled Faces in the Wild)  is a well-known public benchmark for face verification. 
By employing large-scale datasets (e.g. `MS1M-V2` or `MS1M-V3`) and ArcFace, we can simply achieve the state-of-the-art performance (99.83) on `LFW`. 
Here, we give an example about how to test the accuracy on a subset of `LFW`.

```{.python .input  n=8}
def test_face_11(data_set):
    import numpy
    data_list = data_set[0]
    issame_list = data_set[1]
    embeddings = None
    for i in range( len(data_list) ):
        data = data_list[i]
        cx = np.zeros( (batch_size,)+data.shape[1:] )
        ba = 0
        while ba<data.shape[0]:
            bb = min(ba+batch_size, data.shape[0])
            count = bb-ba
            cx[:count,:,:,:] = data[ba:bb, :,:,:]
            xs = gluon.utils.split_and_load(cx, ctx_list=ctx, batch_axis=0)
            embs = [test_net(x).asnumpy() for x in xs]
            embs = numpy.concatenate(embs, axis=0)
            if embeddings is None:
                embeddings = numpy.zeros( (data.shape[0], embs.shape[1]) )
            embeddings[ba:bb,:] += embs[:count,:]
            ba = bb
    embeddings /= len(data_list)
    xnorm = embeddings*embeddings
    xnorm = numpy.sum(xnorm, axis=1, keepdims=True)
    xnorm = numpy.sqrt(xnorm)
    embeddings /= xnorm
    xnorm = numpy.mean(xnorm)
    emb1 = embeddings[0::2]
    emb2 = embeddings[1::2]
    sim = emb1 * emb2
    sim = numpy.sum(sim, 1)
    thresholds = numpy.arange(0, 1, 0.0025)
    actual_issame = issame_list
    acc_max = 0.0
    thresh = 0.0
    for threshold in thresholds:
        predict_issame = numpy.greater(sim, threshold)
        tp = numpy.sum(numpy.logical_and(predict_issame, actual_issame))
        fp = numpy.sum(numpy.logical_and(predict_issame, numpy.logical_not(actual_issame)))
        tn = numpy.sum(numpy.logical_and(numpy.logical_not(predict_issame), numpy.logical_not(actual_issame)))
        fn = numpy.sum(numpy.logical_and(numpy.logical_not(predict_issame), actual_issame))
    
        tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
        fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
        acc = float(tp+tn)/predict_issame.size
        if acc>acc_max:
            acc_max = acc
            thresh = threshold
    return xnorm, acc_max, thresh

lfw_xnorm, lfw_acc, lfw_thresh = test_face_11(val_set['lfw_small'])
print('LFW-Small Accuracy:', lfw_acc)

```

## Summary

1. Defined training and testing block for ArcFace.
2. Face verification testing after training a model.

## Exercises

1. Train for more epochs and use learning-rate decay. Check the final verification accuracy.
2. Use larger datasets (e.g. `CASIA-Webface`) for training,  and check the verification result on the full `LFW` dataset.
3. Use larger networks (e.g. `FaceResNet-50`) for training, and check the verification accuracy.
4. Use different loss functions (e.g. Softmax, SphereFace and CosFace) for training, and check the verification accuracy.
