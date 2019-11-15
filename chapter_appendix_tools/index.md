# Appendix: Tools for Deep Learning
:label:`chap_appendix_tools`

Looking back to the short history of machine learning starting from 1950s, intellectuals are always innovating new models. However, even though deep learning was introduced to the machine learning community in 1986, it did not emerge until 2012. If we wonder why it took such a long time for this field to become prosperous, we need to route back to the fundamental components of deep learning. 


There are 3 pillars of deep learning: the algorithm, the data, and the computing power. 
On the one hand, the algorithm is never the barrier in the history of deep learning evolution. As we can see, most of the well-known deep learning algorithms were born in 1990s, such as recurrent neural network (RNN) and convolution neural network (CNN). On the other hand, the data and the computing power are the limiting factors. First, deep learning has a huge appetite for datasets. 20 years ago, the MNIST dataset with 60, 000 handwritten digit images was considered a large dataset. For the traditional machine learning algorithms, this size is enough. While the 60, 000 data points may be just an appetizer for deep learning. 


Until 2009, at the birth of the ImageNet, we start observing large scale datasets booming such as ImageNet. ImageNet is a large visual database which contains 14 millions images and is designed for use in visual object recognition software research. Interestingly, back then, researchers did not realize the key role of large scale data in deep learning---the ImageNet authors, Professor Fei-fei Li and her group only received a poster to present their work at the conference of Computer Vision and Pattern Recognition (CVPR). Later, the ImageNet dataset evolved to the ImageNet Challenge, where practitioners fitted novel machine learning models and competed for high accuracy object detection and image classification worldwide. In the first 3 year, machine learning algorithms totally outperformed deep learning ones. Does it mean that only a large scale of dataset is not enough for deep learning to succeed?


Indeed, this led to the other core components of deep learning---the computing power. In the ImageNet 2012 Challenge, a convolution neural network called AlexNet which was trained over two graphics processing units (GPUs), elevated the compute power to a next level. Magically, it won the 2012 competition by achieving an error rate more than $10.8\%$ lower than that of the runner up. This is the first time when researchers enabled to train a neural network using multiple GPUs. As a result, the deep learning trend started emerging.


Putting aside the historical anecdotes, the eagle-eyed amongst us may notice the importance of the computing power in deep learning. Therefore, in this chapter, we will walk you through the essential tools for deep learning, from introducing jupyter notebook in :numref:`sec_jupyter` to empowering you training models on AWS in :numref:`sec_aws`. Besides, if you would like to purchase your own GPUs, we also note down some practical suggestions in :numref:`sec_buy_gpu`. Last but not the least, if you are interested in being a contributor of this book, please follow the instructions in :label:`sec_how_to_contribute`.

```toc
:maxdepth: 2

jupyter
aws
buy-gpu
how-to-contribute
d2l
```

