# Faster R-CNN

Faster R-CNN :cite:`Ren.He.Girshick.ea.2015` is an object detection in the Region-based CNN (R-CNN) family, which improves the speedup over the original R-CNN model :cite:`Girshick.Donahue.Darrell.ea.2014` and the following improved version Fast R-CNN :cite:`Girshick.2015`. Like other models in the R-CNN family, Faster R-CNN is a two-stage detection model. High quality regions (anchor boxes) are proposed using the RPN model described in :numref:`sec_rpn`, then an another model is trained for predicting the bounding boxes and object class categories. 

Let's first import the modules and load the data.

```{.python .input  n=1}
import d2l
from mxnet import autograd, np, npx, gluon, init
from mxnet.gluon import nn

npx.set_np()

batch_size = 32
train_iter, test_iter = d2l.load_data_voc_detection(batch_size, 256, 256)
```

## Model

The model architecture is illustrated in :numref:`fig_faster_r-cnn`. We first extract features using a backbone CNN model. The RPN model shares the same backbone as the faster R-CNN model, therefore these features are inputed into the prediction module of RPN to obtain anchors. These anchors are de-duplicated by using NMS :numref:`sec_anchor`. There is a new layer called RoI (region of interest) pooling, which we will describe shortly. The RoI pooling outputs a feature map with a fixed shape for each anchor from the RPN model. Then two dense layers are used for the object class and bounding box predictions. 

![Faster R-CNN model. ](../img/faster-rcnn.svg)
:label:`fig_faster_r-cnn`

Let's look into the ROI pooling layer in details. The layer is somewhat different from the pooling
layers we have discussed before. In a normal pooling layer, we set the pooling
window, padding, and stride to control the output shape. In an RoI pooling
layer, we can directly specify the output shape of each region (anchor box).
Assume this region has a height of $h$ and a width of $w$, and we want the region output has a height of $h'$ and width of $w'$. 
Then this region is divided into
a grid of sub-regions with the shape $h' \times w$. The size of each
sub-region is about $(h/h') \times (w/w')$. The sub-region height and width
must always be integers and the largest element is used as the output for a
given sub-region. This allows the RoI pooling layer to extract features of the
same shape from RoIs of different shapes.

In :numref:`fig_roi`, we select an $3\times 3$ region as an RoI of the $4 \times
4$ input. For this RoI, we use a $2\times 2$ RoI pooling layer to obtain a
single $2\times 2$ output. When we divide the region into four sub-windows, they
respectively contain the elements 0, 1, 4, and 5 (5 is the largest); 2 and 6 (6
is the largest); 8 and 9 (9 is the largest); and 10.

![$2\times 2$ RoI pooling layer. ](../img/roi.svg)
:label:`fig_roi`

Let's generate the input `X` in :numref:`fig_roi` in a 4D format, with both a height and width of 4 and only a single channel.

```{.python .input  n=3}
X = np.arange(16).reshape(1, 1, 4, 4)
X
```

Each region is expressed as five elements: the image index in the batch this region belongs to and the $x, y$ coordinates of its upper-left and bottom-right corners. The `spatial_scale` can be used to scale the coordinates, with $1$ means no scaling.

```{.python .input  n=4}
rois = np.array([[0, 0, 0, 2, 2], [0, 0, 1, 2, 3]])
Y = npx.roi_pooling(X, rois, pooled_size=(2, 2), spatial_scale=1)
print(Y, "with shape", Y.shape)
```

The output has the shape `(num_regions, num_channels, pool_height, pool_width)`. The previous example shows the case with a single channel. For multiple channels, the $i$-th output channel is obtained using the $i$-th channel of $X$, just like other pooling layers. As you can seen, no matter the region locations and shapes in `rois`, RoI pooling always gives fixed shape output, that simplifies the following predictions. 

The output module for faster R-CNN is implemented in `FasterRCNNOutput`. Comparing to `RPNOutput` defined in :numref:`sec_rpn`, here we use dense layers for prediction, as each anchor box has been treated as an example by the RoI pooling layer.

```{.python .input  n=5}
class FasterRCNNOutput(nn.Block):
    def __init__(self, num_classes, **kwargs):
        super(FasterRCNNOutput, self).__init__(**kwargs)
        self.cls_predictor = nn.Dense(num_classes+1)
        self.box_predictor = nn.Dense(4)

    def forward(self, X, rois):
        batch_size, _, w, h = X.shape
        rois[:,1:] *= np.array([[w, h, w, h]], ctx=rois.context)
        Y = npx.roi_pooling(X, rois, pooled_size=(2, 2), spatial_scale=1)
        cls_preds = self.cls_predictor(Y)
        box_preds = self.box_predictor(Y)
        return cls_preds, box_preds
```

## Anchor Boxes

The anchor boxes used by faster R-CNN are the predictions of RPN. More specifically, they are the output of the `predict_rpn` function (:numref:`sec_rpn`) with a 0.7 threshold for NMS. For each ground truth bounding box, anchors with IoU larger than 0.5 are labeled as positive, and negative otherwise. The following function `faster_rcnn_targets` accepts the ground truth bounding boxes in a batch with their RPN predictions, then returns the anchors in the format for the RoI pooling layer, and anchor labels with offsets.

```{.python .input  n=37}
def faster_rcnn_targets(ground_truth, rpn_preds):
    rois, cls_labels, box_offsets = [], [], []
    for i, (label, (_, anchors)) in enumerate(zip(ground_truth, rpn_preds)):
        labels, offsets = d2l.label_anchors(
            np.expand_dims(label, axis=0), anchors, 0.5, 0.5, 5)
        rois.append(np.concatenate((np.ones((len(labels[0]), 1))*i, 
                                    anchors.T), axis=1))
        cls_labels.append(labels[0])
        box_offsets.append(offsets[0])
    return [np.concatenate(x) for x in [rois, cls_labels, box_offsets]]
```

## Training

The faster R-CNN model has three components: the backbone, the RPN output module and the faster R-CNN output module. The original paper recommended the following training strategy :cite:`Ren.He.Girshick.ea.2015` in 4 steps:

1. Train the backbone with the RPN output as :numref:`sec_rpn`.
2. Train the backbone with the faster R-CNN output with the backbone parameters obtained from step 1. 
3. Repeat step 1 starting with the current model parameters.
4. Repeat step 2 starting with the current model parameters. 

Here we just implement the first two steps as we are dealing a simpler dataset. 
Let's first train step 1, where parameters in both `backbone` and `rpn_output` are randomly initialized. We save the backbone parameters on disk to easy hyper-paramter tuning later on.

```{.python .input  n=33}
ctx = d2l.try_gpu()
loss = d2l.DetectionLoss(1)

backbone = d2l.detection_backbone()
backbone.initialize(init=init.Xavier(), ctx=ctx, force_reinit=True)
rpn_output = d2l.RPNOutput()
rpn_output.initialize(init=init.Xavier(), ctx=ctx)

d2l.train_detection(backbone, rpn_output, d2l.rpn_batch, 
                    train_iter, test_iter, loss, num_epochs=3, 
                    backbone_lr=0.5, output_lr=0.5, ctx=ctx)
backbone.save_parameters('backbone')
```

Next we define the batch function for faster R-CNN. It first runs `predict_rpn` to predict the anchor boxes, then obtain the targets with `faster_rcnn_targets`. Note that we don't need to compute the gradients for the RPN output module, while it has been attached gradients on step 1. We put these two steps in the `pause` scope of `autograd` to explicit ask MXNet to stop computing gradients. Besides, as the anchors change for every batch, we ignore `anchors` input and output. 

```{.python .input  n=38}
def faster_rcnn_batch(ground_truths, backbone_outputs, faster_rcnn_output, anchors):
    with autograd.pause():
        rpn_preds = d2l.predict_rpn(backbone_outputs, rpn_output, 
                                    d2l.rpn_anchors, nms_threshold=0.7)
        rois, cls_labels, box_labels = faster_rcnn_targets(
            ground_truths, rpn_preds)
    preds = faster_rcnn_output(backbone_outputs, 
                               rois.copyto(backbone_outputs.context))
    return preds, (cls_labels, box_labels), None
```

Now we can do step 2. We randomly initialize the faster R-CNN output module, but load the parameters of the backbone trained in step 1 from disk. This extra loading step makes we can run step 2 repeated to tune the learning rate that changes the parameters in `backbone`. 

```{.python .input  n=15}
faster_rcnn_output = FasterRCNNOutput(2)
faster_rcnn_output.initialize(init=init.Xavier(), ctx=ctx)
backbone.load_parameters('backbone', ctx=ctx)

d2l.train_detection(backbone, faster_rcnn_output, faster_rcnn_batch, 
                    train_iter, test_iter, loss, num_epochs=3, 
                    backbone_lr=0.01, output_lr=0.1, ctx=ctx)
```

## Summary

1. 

## Exercises

1. Tune both number of epochs and the learning rates for better results. 
1. Implement the full 4-step training procedure
1. Implement the inference function and visualize the results. 

