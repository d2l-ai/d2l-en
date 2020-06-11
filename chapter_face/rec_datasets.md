# Datasets and Pre-processing
:label:`sec_facerecognition_dataset`

Face recognition datasets consist of face images and their corresponding identity labels, which are similar to the traditional classification datasets 
(e.g. CIFAR10/100 and ImageNet). Recently, many public available face recognition datasets have been collected, pre-processed and used to train face recognition models.

```{.python .input  n=1}
import collections
from d2l import mxnet as d2l
import mxnet as mx
from mxnet import autograd, gluon, init, np, npx
import os
import random
import time
import zipfile
import glob


npx.set_np()
```

All our face recognition datasets share the same format:

1. The `CASIA-Webface` dataset contains about 0.5M images of 10K identities.
2. The `CASIA-Webface-Small` dataset, a subset of `CASIA-Webface`, contains about 50K images of 1K identities.
3. The `CASIA-Webface-UltraSmall` dataset, a smaller subset of `CASIA-Webface`, contains about 15K images of 0.3K identities.
4. The `MS1M-V2` dataset, which is cleaned from the MS-Celeb-1M dataset and used in ArcFace, consists of 5.8M images from 85K identities.
5. The `MS1M-V3` dataset, which is used in the lightweight face recognition challenge, includes 5.1M images of 91K identities. All the face images are detected and aligned by the state-of-the-art face detector, RetinaFace.

We can easily read the meta-information (e.g. normalised crop size and identity number) about the above datasets:

```{.python .input  n=2}
# Saved in the d2l package for later use


d2l.DATA_HUB['faces-casia-small'] = (
    'http://d2l-data.s3-accelerate.amazonaws.com/'
    'faces-casia-small.zip', 'b2eed95fd98296c25af65623b8aaa3ee87f982ff')

d2l.DATA_HUB['faces-casia-ultrasmall'] = (
    'http://d2l-data.s3-accelerate.amazonaws.com/'
    'faces-casia-ultrasmall.zip', '352b9a33ca089307a07cec80c23b1765d304e95a')
    

# Saved in the d2l package for later use
def _read_facerec_meta(data_dir):
    file_name = os.path.join(data_dir, 'property')
    line = open(file_name, 'r').readlines()[0]
    vals = [int(x) for x in line.split(',')]
    return (vals[1], vals[2]), vals[0]

dataset = "faces-casia-ultrasmall"
data_dir = d2l.download_extract(dataset, dataset)
print(_read_facerec_meta(data_dir))

```

## Dataset Visualization

There are 300 persons in the `CASIA-Webface-UltraSmall` dataset, let's check how images look like in this dataset.

To get all images by a specific person ID:

```{.python .input  n=3}

def _facerec_vis_id(data_dir, person_id):
    folder = os.path.join(data_dir, "%08d"%person_id)
    image_path_list = glob.glob(os.path.join(folder, '*.jpg'))
    img_list = []
    for image_path in image_path_list:
        img = npx.image.imread(image_path).asnumpy()[:,:,::-1]
        #print(img.__class__)
        #print(img.shape)
        img_list.append(img)
    return img_list
     
def _show_imlist(img_list, max_num=10):
    import numpy
    if len(img_list)>max_num:
        img_list = img_list[:max_num]
    im = numpy.concatenate(img_list, axis=1)[:,:,::-1]
    d2l.plt.imshow(im)
    d2l.plt.show()
    
_img_list = _facerec_vis_id(data_dir, 19)
_show_imlist(_img_list)
_img_list = _facerec_vis_id(data_dir, 209)
_show_imlist(_img_list)
```

## Data Loader

Finally, we define the data loader. For training face recognition models, only random-flip is used as data augmentation.
For fast validation, we also provide some 1:1 face verification test sets. During training, they will be tested periodically.

```{.python .input  n=4}

from mxnet.gluon.data import Dataset
from mxnet.gluon.data.vision import transforms

class FaceDataset(Dataset):
    def __init__(self, data_dir):
        
        image_size, num_classes = _read_facerec_meta(data_dir)
        self.transform = transforms.Compose([
            transforms.RandomFlipLeftRight(),
            transforms.ToTensor()
        ])

        self.seq = []
        for i in range(num_classes):
            id_dir = os.path.join(data_dir, "%08d"%i)
            image_path_list = glob.glob(os.path.join(id_dir, "*.jpg"))
            for image_path in image_path_list:
                self.seq.append( (image_path, i))

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        item = self.seq[idx]
        image = npx.image.imread(item[0])
        label = item[1]
        image = self.transform(image)
        return image, label

def load_data_face_rec(data_dir, load_val=[], batch_size=512):
    
    ds = FaceDataset(data_dir) 
    loader = gluon.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers = d2l.get_dataloader_workers(), last_batch='discard')
    val_set = {} #ignore here
    return loader, val_set

batch_size = 512
loader, _ = load_data_face_rec(data_dir, [], batch_size)
for batch_idx, (data, label) in enumerate(loader):
    print(data.shape)
    print(label.shape)
    assert data.shape==(batch_size, 3, 112, 112)
    assert label.shape==(batch_size,)
    break
```

## Normalised Face Crop

Facial images in the above datasets are already pre-processed for training. 
The following `_face_align` function gives details about generating the normalised face crop. 
Given the five facial landmarks (eye corners, nose tip, and mouth corners) on the original face image, we estimate an affine transform matrix to generate a normalised face crop using a fixed landmark template (`arcface_dst`) defined on the mean face. The size of the normalised crop is 112x112 (`image_size`).

```{.python .input  n=5}
# Saved in the d2l package for later use
def _face_align(image, landmark):
    import cv2
    from skimage import transform as trans
    tform = trans.SimilarityTransform()
    assert landmark.shape==(5,2)
    image_size = 112
    arcface_dst = np.array([
      [38.2946, 51.6963],
      [73.5318, 51.5014],
      [56.0252, 71.7366],
      [41.5493, 92.3655],
      [70.7299, 92.2041] ], dtype=np.float32 )
    tform.estimate(landmark, arcface_dst)
    M = tform.params[0:2,:]
    aligned = cv2.warpAffine(image,M, (image_size, image_size), borderValue = 0.0)
    return aligned
```

## Summary

* Introduction of common face recognition datasets.
* How to visualize and read datasets.
* How to generate normalised face crops by using facial landmarks.
