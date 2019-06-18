# Installation
:label:`chapter_installation`

To get you up and running with hands-on experiences, we'll need you to set up with a Python environment, Jupyter's interactive notebooks, the relevant libraries, and the code needed to *run the book*.

## Obtaining Source Codes

The source code package containing all notebooks is available at https://d2l.ai/d2l-en.zip. Please download it and extract it into a folder. For example, on Linux/macos, if you have both `wget` and `unzip` installed, you can do it through:  

```
wget https://d2l.ai/d2l-en.zip
unzip d2l-en.zip -d d2l-en
```

## Installing Running Environment

If you have both Python 3.5 or older and pip installed, the easiest way to install the running environment through pip. Two packages are needed, `d2l` for all dependencies such as Jupyter and saved code blocks, and `mxnet` for deep learning framework we are using. First install `d2l` by

```bash
pip install d2l
```

If unfortunately something went wrong, please check

1. You are using `pip` for Python 3 instead of Python 2 by checking `pip --version`. If it's Python 2, then you may check if there is a `pip3` available.
2. You are using a recent `pip`, such as version 19. Otherwise you can upgrade it through `pip install --upgrade pip`
3. If you don't have permission to install package in system wide, you can install to your home directory by adding a `--user` flag. Such as `pip install d2l --user`

Before installing `mxnet`, please first check if you are able to access GPUs. If so, please go to :ref:`sec_gpu` for instructions to install a GPU-supported `mxnet`. Otherwise, we can install the CPU version, which is still good enough for the first few chapters.  

```bash
pip install mxnet
```

Once both packages are installed, we now open the Jupyter notebook by

```bash
jupyter notebook
```

At this point open http://localhost:8888 (which usually opens automatically) in the browser, then you can view and run the code in each section of the book.

## Upgrade to a New Version

Both this book and MXNet are keeping improving. You may want to check a new version from time to time. 

1. This URL  https://d2l.ai/d2l-en.zip always points to the contents. 
2. You can upgrade `d2l` by `pip install d2l -U` or even just install the latest version from Github by `pip install git+https://github.com/d2l-ai/d2l-en`.  
3. MXNet can be upgraded by `pip install MXNet -U` as well. 

## GPU Support

:label:`sec_gpu`

By default MXNet is installed without GPU support to ensure that it will run on any computer (including most laptops). Part of this book requires or recommends running with GPU. If your computer has NVIDIA graphics cards and has installed [CUDA](https://developer.nvidia.com/cuda-downloads), you should install a GPU-enabled MXNet. 

If you have installed the CPU-only version, then remove it first by

```bash
pip uninstall mxnet
```

Then you need to find the CUDA version you installed. You may check it through `nvcc --version` or `cat /usr/local/cuda/version.txt`. Assume you have installed CUDA 10.1, then you can install the according MXNet version by 

```bash
pip install mxnet-cu101
```

You may change the last digits according to your CUDA version, e.g. `cu100` for CUDA 10.0 and `cu90` for CUDA 9.0. You can find all available MXNet versions by `pip search mxnet`. 

## Exercises

1. Download the code for the book and install the runtime environment.


## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2315)

![](../img/qr_install.svg)
