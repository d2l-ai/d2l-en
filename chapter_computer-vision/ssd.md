# Single Shot Multibox Detection (SSD)

In both RPN (:numref:`sec_rpn`) and faster R-CNN (:numref:`sec_rcnn`), the backbone model outputs $8\times 8$ feature maps for the images. If an object in the image with height and width less than 1/8 of the image height and width, then it is presented as a single pixel in the feature maps, which makes detection difficult. We can increase the input image resolution and reduce the backbone model stride to improve the detection accuracy for small objects. In this chapter, we will introduce single shot multibox detection
(SSD) :cite:`Liu.Anguelov.Erhan.ea.2016`, which uses the internal feature maps in the backbone model for multiple scale detection. Unlike faster R-CNN, it only has a single stage, that's why it is called "single shot".  

Let's first import modules and load the dataset. 

```{.python .input  n=1}
from mxnet import autograd, np, npx, gluon, init
from mxnet.gluon import nn
import d2l

npx.set_np()

batch_size = 32
train_iter, test_iter = d2l.load_data_voc_detection(batch_size, 256, 256)
```

## Model

:numref:`fig_ssd` shows the design of an SSD model. It is similar to RPN (:numref:`sec_rpn`) on generating anchor boxes and predictions, despite that SSD predicts box categories instead of the binary prediction in RPN. The main difference is that not only the backbone output feature maps are used for prediction. Both internal feature maps of the backbone model, which have larger width and height, and the output of a global max pooling, are used for prediction as well. 

![The SSD is composed of a base network block and several multiscale feature blocks connected in a series. ](../img/ssd.svg)
:label:`fig_ssd`

Remember the ResNet-18 model contains multiple convolution blocks (:numref:`sec_resnet`), let's check the output size of each block. 

```{.python .input  n=3}
for X, Y in train_iter:
    break
net = d2l.detection_backbone()
for i in range(len(net)):
    X = net[i](X)
    print('layer', i, ':', X.shape)
```

As can be seen, each block either keep the input size or halve the width and height while double the channel size. 
In our SSD implementation in `SSDBackbone`, we will use 4 feature maps for object detection. The first layer is the output of the 6-th block (layer 6), which outputs a $32\times 32$ feature map. The second layer is the output of the 7-th block, with a $16\times 16$ feature map. The third layer is the output of the backbone model, with a $8\times 8$ feature map. The last layer is a $1\times 1$ feature map by applying a global max pooling. The output is a list of the feature map of each layer. 

```{.python .input  n=4}
class SSDBackbone(nn.Block):
    def __init__(self, **kwargs):
        super(SSDBackbone, self).__init__(**kwargs)
        net = d2l.detection_backbone()
        self.backbone = nn.Sequential()
        self.backbone.add(net[0:7])
        self.backbone.add(net[7])
        self.backbone.add(net[8:10])
        self.backbone.add(nn.GlobalMaxPool2D())
        
    def forward(self, X):
        rets = []
        for blk in self.backbone:
            X = blk(X)
            rets.append(X)
        return rets
```

Let's verify the feature map shapes of the 4 layers.

```{.python .input  n=5}
for X, Y in train_iter:
    break
    
backbone = SSDBackbone()
feature_maps = backbone(X)

for i, x in enumerate(feature_maps):
    print('layer', i, ':', x.shape)
```

The output module of SSD is also similar to RPN, despite that we create two $1\times 1$ convolution layers for each feature map layer, and the class prediction layer predicts object categories.

```{.python .input  n=8}
class SSDOutput(nn.Block):
    def __init__(self, num_classes, **kwargs):
        super(SSDOutput, self).__init__(**kwargs)
        self.cls_predictors = nn.Sequential()
        self.box_predictors = nn.Sequential()
        self.num_classes = num_classes + 1  # add background class
        self.num_anchors = 4  # defined by function ssd_anchors
        for i in range(4):
            self.cls_predictors.add(nn.Conv2D(
                self.num_anchors*self.num_classes, kernel_size=3, padding=1))
            self.box_predictors.add(nn.Conv2D(
                self.num_anchors*4, kernel_size=3, padding=1))
        
    def forward(self, feature_maps):
        cls_preds, box_preds = [], []
        for X, cls, box in zip(feature_maps, self.cls_predictors, 
                               self.box_predictors):
            cls_preds.append(
                cls(X).transpose(0,2,3,1).reshape(-1, self.num_classes))
            box_preds.append(box(X).transpose(0,2,3,1).reshape(-1, 4))
        return (np.concatenate(cls_preds, axis=0), 
                np.concatenate(box_preds, axis=0))
```

The outputs should have two ndarrays with shape `(batch_size*num_anchors, num_classes+1)` and `(batch_size*num_anchors, 4)`, where `num_anchors` will be defined next.

```{.python .input  n=9}
ssd_output = SSDOutput(2)
ssd_output.initialize()
cls_preds, box_preds = ssd_output(feature_maps)
cls_preds.shape, box_preds.shape
```

## Anchor Boxes

For each layer, we generate anchor boxes by calling `generate_anchors` defined in :numref:`sec_anchors`. Note that the anchor sizes for each layer are different. The smallest size is $0.3$, then we increase it by $0.1$ each time. So that low layers uses small sizes for small objects, while high layers for large objects. The anchors for each layers are then concatenated. 

```{.python .input  n=6}
# Saved in the d2l package for later use
def ssd_anchors(feature_maps):
    sizes = [[.3,.4],[.5,.6],[.7,.8],[.9,.1]]
    #[[0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
             #[0.88, 0.961]]    
    ratios = [[1, 2, 0.5]] * 4
    anchors = []
    for X, s, r in zip(feature_maps, sizes, ratios):
        _, _, height, width = X.shape
        anchors.append(d2l.generate_anchors(height, width, s, r))
    return np.concatenate(anchors, axis=1)
```

Comparing RPN, more anchor boxes are used in SSD.

```{.python .input  n=7}
anchors = ssd_anchors(feature_maps)
anchors.shape
```

And also there are more positive anchors.

```{.python .input  n=10}
cls_labels, _ = d2l.rpn_targets(Y, anchors)
print('positive:', (cls_labels==1).sum())
print('negative:', (cls_labels==0).sum())
```

## Training

The batch function is similar to `rpn_batch` (:numref:`sec_rpn`) except for using `ssd_anchors` to generate anchors.

```{.python .input}
def ssd_batch(Y, backbone_features, output_model, anchors):
    if anchors is None:
        anchors = ssd_anchors(backbone_features)
    labels = d2l.rpn_targets(Y, anchors)
    preds = output_model(backbone_features)
    return preds, labels, anchors
```

The training is almost identical to RPN. 

```{.python .input  n=18}
ctx, loss = d2l.try_gpu(), d2l.DetectionLoss(1)
backbone = SSDBackbone()
backbone.initialize(init=init.Xavier(), ctx=ctx, force_reinit=True)
ssd_output = SSDOutput(2)
ssd_output.initialize(init=init.Xavier(), ctx=ctx)

lr = 0.05
num_epochs = 4
d2l.train_detection(backbone, ssd_output, ssd_batch, train_iter, 
                    test_iter, loss, num_epochs, lr, lr, ctx)
```

## Inference

We can call `predict_rpn` directly for predictions, despite it always use the confident scores for the first object category. 

```{.python .input  n=20}
for X, Y in test_iter:
    break 

preds = d2l.predict_rpn(backbone(X.copyto(ctx)), ssd_output, ssd_anchors, 0.7)
```

```{.python .input}
def visualize_rpn_preds(X, rpn_preds):
    imgs = [img.transpose(1, 2, 0)*d2l.RGB_STD+d2l.RGB_MEAN for img in X[:10]]
    axes = d2l.show_images(imgs, 2, 5, scale=2)
    for ax, label in zip(axes, rpn_preds[:10]):
        h, w, _ = imgs[0].shape
        scores = ['%.1f'%i for i in label[0][:2]]
        boxes = label[1][:,:1].T*np.array([w,h,w,h])
        d2l.show_boxes(ax, boxes, colors=['w'], labels=scores)
        
visualize_rpn_preds(X, preds)
```

## Summary

