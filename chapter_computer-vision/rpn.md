# Region Proposal Networks
:label:`sec_rpn`

Region Proposal Networks (RPN) is not an object detection algorithm, but a module used in faster RCNN to propose high-quality anchors boxes :cite:`Ren.He.Girshick.ea.2015`. As RPN is simple and contains all components for an anchor-based object detection algorithm, we present it in this section before diving into more complicated models.

```{.python .input  n=1}
import d2l
from mxnet import autograd, np, npx, gluon, init, image
from mxnet.gluon import nn

npx.set_np()
```

In the original paper, images are resized so that its short edge is 600px with the aspect ratio unchanged. Using such high resolution images makes small objects easy to detect, but increases the computation cost. Also as images may have different shapes due to keeping the aspect ratio, RPN can only use a batch size of 1 during training that leads to computational inefficiency. To make the training fast, we resize the images into $256\times 256$ with a batch size of 32. It degrades the model accuracy, but is good enough as a demo purpose.

```{.python .input  n=2}
batch_size = 32
train_iter, test_iter = d2l.load_data_voc_detection(batch_size, 256, 256)
```

## Model

In :numref:`sec_anchor`, we saw how to generate a set of anchor boxes centered at every pixel to cover every possible object locations. It leads to a large amount of anchors that increases the computational cost. RPNs aims to reduce the number of anchor boxes by predicting on the input image. It's similar to a normal object detection model that predicts the object locations. One main difference is that RPN only performs a binary classification to predict if or not a location contains an object, while an object detector needs to predict the object class categories. 

The model architecture of RPN is illustrated in :numref:`fig_rpn`. We first use a CNN backbone to compute the feature maps of an input image. We generate anchor boxes and label them by the ways introduced in :numref:`sec_anchor` based on the width and height of the backbone features. These features are then inputed into a convolution layer followed by two parallel layers to obtain classification predictions and bounding box offset predictions. 

![Mode architecture for RPN](../img/rpn.svg)
:label:`fig_rpn`

Let's first implement the backbone model. In the original paper, a VGG-16 model (:numref:`sec_vgg`) is used as the backbone. Here we use the more modern `resnet18_v2` as :numref:`sec_fine_tuning`. Remember the last two layers in the `feature` module are the global averaging pooling layer and the flatten layer, we remove both of them as we need the feature maps before the last pooling layer.

```{.python .input  n=3}
# Saved in the d2l package for later use
def detection_backbone():
    net = gluon.model_zoo.vision.resnet18_v2(pretrained=True)
    return net.features[:-2]
```

Let's input the first image batch into the model. We can see that this backbone model transfers an $256\times 256$ image into a $8\times 8$ feature map, with a stride of 32. In :cite:`Ren.He.Girshick.ea.2015`, its VGG-16 backbone has a stride of 16 to obtain a $16\times 16$ feature map. A larger feature map benefits detecting small objects, while leads to more computation cost as more anchor boxes will be generated.

```{.python .input  n=4}
for X, Y in train_iter:
    break

backbone = detection_backbone()
X = backbone(X)
X.shape
```

The convolution layer with the two prediction layers after the backbone are implemented in the class `RPNOutput`. The convolution layer is chosen with a $3\times 3$ kernel with 1 padding so that that it doesn't not change the feature map size while reducing the channel size from 512 to 128. 

We use $1\times 1$ convolution layers for predictions. If such a layer has $n$ channels, then it makes $n$ prediction for every pixel in the backbone features. Assume we generate $k$ anchor boxes for each pixel, then the output channel size of the classification layer should be $2k$, and the for the bounding box offsets layer should be $4k$.  we should allocate $2k$ channels. Lastly, the predictions are reshaped into 3D arrays to compute the loss.

```{.python .input  n=5}
# Saved in the d2l package for later use
class RPNOutput(nn.Block):
    def __init__(self, num_anchors=5, **kwargs):
        super(RPNOutput, self).__init__(**kwargs)
        self.conv = nn.Conv2D(128, kernel_size=3, padding=1, activation='relu')
        self.cls_predictor = nn.Conv2D(num_anchors*2, kernel_size=1)
        self.box_predictor = nn.Conv2D(num_anchors*4, kernel_size=1)

    def forward(self, X):
        Y = self.conv(X)
        cls_preds = self.cls_predictor(Y)
        box_preds = self.box_predictor(Y)
        return (cls_preds.transpose(0, 2, 3, 1).reshape(-1, 2),
                box_preds.transpose(0, 2, 3, 1).reshape(-1, 4))
```

If the feature map has a $w$ width and a $h$ height, then the outputs should have two ndarrays with shape `(batch_size*n*w*h, 2)` and `(batch_size*n*w*h, 4)`, respectively.

```{.python .input  n=6}
rpn_output = RPNOutput()
rpn_output.initialize()
preds = rpn_output(X)
preds[0].shape, preds[1].shape
```

## Anchor Boxes

In :numref:`sec_anchors`, we discussed how to sample anchor boxes and predict based on offsets. In the following, `rpn_anchors` generates anchors with three sizes and aspect ratios.

```{.python .input  n=7}
# Saved in the d2l package for later use
def rpn_anchors(X):
    """Generate anchors for RPN"""
    sizes, ratios = [0.85, 0.43, 0.21], [1, 2, 0.5]
    _, _, height, width = X.shape
    return d2l.generate_anchors(height, width, sizes, ratios)
```

Now let's look at training. During training, each anchor box is treated as a training example. As these anchor boxes are generated, we need to assign labels to them for training. The label has two parts. First, we mark if an anchor box contains an object, or only has background, or should be ignored. Second, if containing an object, we compute the offset from this anchor box to the ground-truth bounding box if this object.


In FPN, we assign three categories to each anchor box: $1$ for positive anchors that containing an object, $0$ negative anchors that have no object, $-1$ for invalid anchors. Given an image, assume it has ground-truth bounding boxes $g_1, \ldots, g_n$, and anchor boxes $a_1, \ldots, a_m$. For each ground-truth $g_i$, we mark $a_j$ as containing an object if the IoU between $g_i$ and $a_j$ is larger than 0.7. If no such $g_i$ exists, we mark the one has the largest IoU to $g_i$. An anchor box $a_j$ is marked as only containing background if there is no ground-truth $g_i$ such as the IoU between $a_j$ and $g_i$ is greater than $0.3$. The rest anchor boxes are marked as invalid. If we generated too much positive or negative anchor boxes for each image, we can randomly sample some of them.

Function `rpn_targets` implements the above labeling strategy, where we keep all positive anchors while sample 100 negative anchors for each image.

```{.python .input  n=8}
# Saved in the d2l package for later use
def rpn_targets(ground_truth, anchors):
    """Assign labels to anchors for RPN
    """
    cls_labels, box_offsets = d2l.label_anchors(
        ground_truth, anchors, 0.7, 0.3, 5)
    cls_labels[cls_labels>0] = 1
    return cls_labels.reshape(-1), box_offsets.reshape(-1, 4)
```

Check the number of positive and negative anchors in the first batch.

```{.python .input  n=9}
anchors = rpn_anchors(X)
cls_labels, _ = rpn_targets(Y, anchors)
print('positive:', (cls_labels==1).sum())
print('negative:', (cls_labels==0).sum())
```

We can also visualize the positive anchors (even columns) and negative anchors (odd columns).

```{.python .input  n=10}
d2l.set_figsize()
axes = d2l.show_images(np.ones((18, 100, 100, 3)), 3, 6)
for label in [0, 1]:
    for ax, gt, cls in zip(axes[label::2], Y[:9], 
                           cls_labels.reshape(X.shape[0], -1)[:9]):
        d2l.show_boxes(ax, gt[gt[:,0]>=0][:,1:5]*100, colors=['k'])
        d2l.show_boxes(ax, anchors[:,(cls==label).nonzero()[0]].T*100,
                       colors=['r' if label == 0 else 'b'])
```

## Loss and Evaluation Functions

The loss has two parts, one is the class prediction loss, and the other one is the bounding box offset prediction loss. The former is a standard classification problem so we can use the cross entropy loss. The later is a regression problem. Instead of the squared loss, here we use $\ell_1$ loss, i.e. $\ell_1(\mathbf a, \mathbf b) = \frac 1n\sum_{i=1}^n |a_i-b_i|$,
to penalty less for large errors. These two loss values are summed with a $\lambda$ that trade-off classification error and bounding box regression error. We often chose a value such as these two terms have a rough equal value.

In addition, we ignore invalid anchors in the classification loss, and only consider positive anchors in the regression loss. Both can be done through the weight argument.

```{.python .input  n=11}
# Saved in the d2l package for later use
class DetectionLoss(nn.Block):
    """Object class loss + lambda * bounding box loss"""
    def __init__(self, lambd, **kwargs):
        super(DetectionLoss, self).__init__(**kwargs)
        self.cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()
        self.box_loss = gluon.loss.L1Loss()
        self.lambd = lambd

    def forward(self, cls_preds, box_preds, cls_labels, box_labels):
        """Compute loss

        cls_preds: (#anchors, #classes)
        box_preds: (#anchors, 4)
        cls_labels: (#anchors, )
        box_labels: (#anchors, 4)
        """
        # Ignore invalid examples
        cls_mask, box_mask = cls_labels >= 0, cls_labels > 0
        cls = self.cls_loss(cls_preds, cls_labels,
                            np.expand_dims(cls_mask, axis=-1))
        box = self.box_loss(box_preds, box_labels,
                            np.expand_dims(box_mask, axis=-1))
        # Average per each image
        cls = cls / float(cls_mask.sum())
        box = box / float(box_mask.sum())
        return cls + self.lambd * box
```

Evaluation is similar to loss. The following class accept the same argument as before, but the classification accuracy is used instead of the cross entropy loss, and returns four elements so we can cumulate them later.

```{.python .input  n=12}
# Saved in the d2l package for later use
class DetectionEvaluation(object):
    def __call__(self, cls_preds, box_preds, cls_labels, box_labels):
        cls_mask, box_mask = cls_labels >= 0, cls_labels > 0
        cls = (cls_preds.argmax(axis=-1) == cls_labels)[cls_mask].sum()
        box = np.abs(box_labels - box_preds)[box_mask, :].sum() / 4
        return cls, box, cls_mask.sum(), box_mask.sum()
```

## Training



We define a function `train_detection` that is able to train the detection models introduced in this book. It's similar to the training functions we implemented before, but here the model has two parts: the backbone and the output model, with two learning rates for each of them. It also accepts a customized function `batch_fn` to computes the predictions, data labels and anchors.

```{.python .input  n=14}
# Saved in the d2l package for later use
def train_detection(backbone, output_model, batch_fn, 
                    train_iter, test_iter, loss, num_epochs, 
                    backbone_lr, output_lr, ctx=d2l.try_gpu()):
    """Train a detection model
    
    backbone, output_model : neural network models 
    batch_fn : a function to compute predictions, labels and anchors
    train_iter, test_iter - data iterators for training and test
    num_epochs - number of data epochs to train
    backbone_lr, output_lr - the learning rates
    """
    backbone_trainer = gluon.Trainer(backbone.collect_params(), 'sgd', {
        'learning_rate': backbone_lr, 'wd': 1e-4})
    output_trainer = gluon.Trainer(output_model.collect_params(), 'sgd', {
        'learning_rate': output_lr, 'wd': 1e-4})
    evaluator, anchors = DetectionEvaluation(), None
    animator = d2l.Animator(xlabel='epoch', xlim=[0,num_epochs], 
                            legend=['train class err', 'train box mae',
                                    'test class err', 'test box mae'])
    for epoch in range(num_epochs):
        # accuracy_sum, mae_sum, num_valid_anchors, num_pos_anchors
        metric = d2l.Accumulator(5)
        for i, (X, Y) in enumerate(train_iter):
            with autograd.record():
                X = backbone(X.as_in_context(ctx))
                preds, labels, anchors = batch_fn(Y, X, output_model, anchors)
                labels = [y.as_in_context(ctx) for y in labels]
                l = loss(*preds, *labels)
            l.backward()
            output_trainer.step(1)
            backbone_trainer.step(1)
            metric.add(*evaluator(*preds, *labels))
            train_err, train_mae = 1-metric[0]/metric[2], metric[1]/metric[3]
            if (i+1)%4 == 0:
                animator.add(epoch+i/len(train_iter), (
                    train_err, train_mae, None, None))
        metric.reset()
        test_err, test_mae = 0, 0  # fixme, remove it later
        continue                   # fixme, remove it later
        for X, Y in test_iter:
            X = backbone(X.as_in_context(ctx))
            preds, labels, anchors = batch_fn(Y, X, output_model, anchors)
            labels = [y.as_in_context(ctx) for y in labels]
            metric.add(*evaluator(*preds, *labels))
        test_err, test_mae = 1-metric[0]/metric[2], metric[1]/metric[3]
        animator.add(epoch+1, (None, None, test_err, test_mae))
    print('train class err %.2f, box mae %.2f; test class err %.2f, '
          'box mae %.2f' % (train_err, train_mae, test_err, test_mae))
```

The batch function for RPN is defined as following. As the data shape is fixed, we only compute the anchors at the first time. Then we label the anchor boxes with `rpn_anchors` and make predictions with the output module `output_model`.

```{.python .input  n=13}
# Saved in the d2l package for later use
def rpn_batch(Y, backbone_features, output_model, anchors):
    if anchors is None:
        anchors = rpn_anchors(backbone_features)
    labels = rpn_targets(Y, anchors)
    preds = output_model(backbone_features)
    return preds, labels, anchors
```

We train a model with 4 data epochs. Note that we train the backbone model from scratch instead of fine tuning.

```{.python .input  n=16}
ctx, loss = d2l.try_gpu(), DetectionLoss(1)

backbone = detection_backbone()
backbone.initialize(init=init.Xavier(), ctx=ctx, force_reinit=True)
rpn_output = RPNOutput()
rpn_output.initialize(init=init.Xavier(), ctx=ctx)

train_detection(backbone, rpn_output, rpn_batch, train_iter, test_iter, loss,
                num_epochs=5, backbone_lr=.5, output_lr=.5, ctx=ctx)
```

## Inference

During inference, we need two additional steps. One is apply the predicted offsets to the anchor boxes to obtain the predicted bounding boxes via `anchor_plus_offset`. Then NMS is used to remove duplications according to the prediction scores of the positive class.

```{.python .input  n=25}
# Saved in the d2l package for later use
def predict_rpn(backbone_features, output_model, anchor_fn, nms_threshold):
    anchors = anchor_fn(backbone_features)
    cls_preds, box_preds = output_model(backbone_features)
    # reshape to 3D to iterate on each image in the batch
    num_anchors, ctx = anchors.shape[1], anchors.context
    batch_size = cls_preds.shape[0] // num_anchors
    cls_preds, box_preds = [
        a.reshape(batch_size, num_anchors, -1).as_in_context(ctx) 
        for a in [cls_preds, box_preds]]
    Y = []
    for i, (cls_pred, box_pred) in enumerate(zip(cls_preds, box_preds)):
        box = d2l.anchor_plus_offset(anchors, box_pred.T)
        keep = d2l.nms(cls_pred[:,1], box, nms_threshold, use_numpy=True)
        Y.append([cls_pred[keep,1], d2l.project_box(box[:,keep])])
    return Y
```

Run the prediction on the first batch of the testing dataset with a NMS threshold of 0.5

```{.python .input  n=23}
for X, Y in test_iter:
    break

preds = predict_rpn(backbone(X.copyto(ctx)), rpn_output, rpn_anchors, 0.5)
```

Last, we visualize the predicted bounding boxes versus the ground truth bonding boxes. You can see that the predicted results are reasonable, but not very accurate.

```{.python .input  n=24}
# Saved in the d2l package for later use
def visualize_rpn_preds(X, Y, rpn_preds):
    imgs = [img.transpose(1, 2, 0)*d2l.RGB_STD+d2l.RGB_MEAN for img in X[:10]]
    axes = d2l.show_images(imgs, 2, 5, scale=2)
    for ax, label, gt in zip(axes, rpn_preds[:10], Y[:10]):
        h, w, _ = imgs[0].shape
        scale = np.array([w,h,w,h])
        boxes = gt[gt[:,0]>=0][:,1:5]*scale
        d2l.show_boxes(ax, boxes, colors=['r'], labels=['gt'])
        scores = ['%.1f'%i for i in label[0][:2]]
        boxes = label[1][:,:1].T*scale
        d2l.show_boxes(ax, boxes, colors=['w'], labels=scores)
        
visualize_rpn_preds(X, Y, preds)
```

## Summary

1. Region Proposal Networks (RPN) aims to predict anchor boxes that contains objects. 
1. RPN uses a CNN backbone to extract features, followed by a convolutional layer to reduce the number of channels and two parallel convolutional layers to predict categories and bounding boxes. 

## Exercises

1. Try fine-tune the backbone instead of training from scratch, check if it improves the results.
1. Improve the model accuracy. You may consider the following options: change model hyper-parameters, using more epochs, tuning the learning rate. 
