# Installation
:label:`chap_installation`

In order to get you up and running for hands-on learning experience,
we need to set up you up with an environment for running Python,
Jupyter notebooks, the relevant libraries,
and the code needed to run the book itself.

## Installing Miniconda

The simplest way to get going will be to install
[Miniconda](https://conda.io/en/latest/miniconda.html). The Python 3.x version
is recommended. You can skip the following stesp asdf kthis step if conda has already installed.
Download the corresponding Miniconda "sh" file from the website
and then execute the installation from the command line
using `sudo sh <FILENAME> -b`. For macOS users:

```bash
# The file name is subject to changes
sudo sh Miniconda3-latest-MacOSX-x86_64.sh -b
```

For Linux users:

```bash
# The file name is subject to changes
sudo sh Miniconda3-latest-Linux-x86_64.sh -b
```

Next initialize the shell so we can run `conda` direclty.

```bash
~/miniconda3/bin/conda init
```

Then close and re-open your current shell, you should be able to create a new
environment as following now:

```bash
conda create --name d2l -y
```


## Downloading the d2l Notebooks

Next, we need to download the code for this book. You can click this
[link](http://d2l.ai/d2l-en.zip) to download and then unzip it. Or if you have
both `wget` and `unzip` available, then

```bash
mkdir d2l-en && cd d2l-en
wget https://d2l.ai/d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
```

Now we can activate the "d2l" environment to install necessary libraries:

```bash
conda activate d2l
conda install python=3.7 pip -y
pip install git+https://github.com/d2l-ai/d2l-en@numpy2
```


## Installing MXNet

Before installing `mxnet`, please first check
whether or not you have proper GPUs on your machine
(the GPUs that power the display on a standard laptop
do not count for our purposes.
If you are installing on a GPU server,
proceed to :ref:`sec_gpu` for instructions
to install a GPU-supported `mxnet`.

Otherwise, you can install the CPU version.
That will be more than enough horsepower to get you
through the first few chapters but you will want
to access GPUs before running larger models.

```bash
pip install --pre mxnet
```

Once both packages are installed, we now open the Jupyter notebook by running:

```bash
jupyter notebook
```


At this point, you can open http://localhost:8888 (it usually opens automatically) in your web browser. Once in the notebook server, we can run the code for each section of the book.

## Upgrade to a New Version

Both this book and MXNet are keeping improving. Please check a new version from time to time.

1. The URL https://d2l.ai/d2l-en.zip always points to the latest contents.
2. Please upgrade "d2l" by `pip install -U git+https://github.com/d2l-ai/d2l-en@numpy2`.
3. For the CPU version, MXNet can be upgraded by `pip install -U --pre mxnet`


## GPU Support

:label:`sec_gpu`

By default, MXNet is installed without GPU support
to ensure that it will run on any computer (including most laptops).
Part of this book requires or recommends running with GPU.
If your computer has NVIDIA graphics cards and has installed [CUDA](https://developer.nvidia.com/cuda-downloads),
then you should install a GPU-enabled MXNet.
If you have installed the CPU-only version,
you may need to remove it first by running:

```bash
pip uninstall mxnet
```


Then we need to find the CUDA version you installed.
You may check it through `nvcc --version` or `cat /usr/local/cuda/version.txt`.
Assume you have installed CUDA 10.1,
then you can install the latest MXNet version
with the following (command:

```bash
pip install --pre mxnet-cu101
```


You may change the last digits according to your CUDA version,
e.g., `cu100` for CUDA 10.0 and `cu90` for CUDA 9.0.
You can find all available MXNet versions via `pip search mxnet`.

For installation of MXNet on other platforms, please refer to http://numpy.mxnet.io/#installation.


## Exercises

1. Download the code for the book and install the runtime environment.


## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2315)

![](../img/qr_install.svg)
