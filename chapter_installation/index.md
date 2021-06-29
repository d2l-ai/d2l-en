# Installation
:label:`chap_installation`

In order to get you up and running for hands-on learning experience,
we need to set you up with an environment 
for running Python, Jupyter notebooks, the relevant libraries, 
and the code needed to run the book itself.

## Installing Miniconda

The simplest way to get going will be to install
[Miniconda](https://conda.io/en/latest/miniconda.html). 
The Python 3.x version is required. 
You can skip the following steps 
if your machine already has conda installed.

Visit the Miniconda website and determine 
the appropriate version for your system
based on your Python 3.x version and machine architecture.
For example, if you are using macOS and Python 3.x 
you would download the bash script 
whose name contains the strings "Miniconda3" and "MacOSX",
navigate to the download location,
and execute the installation as follows:

```bash
sh Miniconda3-latest-MacOSX-x86_64.sh -b
```


A Linux user with Python 3.x 
would download the file
whose name contains the strings "Miniconda3" and "Linux" 
and execute the following at the download location:

```bash
sh Miniconda3-latest-Linux-x86_64.sh -b
```


Next, initialize the shell so we can run `conda` directly.

```bash
~/miniconda3/bin/conda init
```


Now close and reopen your current shell. 
You should be able to create 
a new environment as follows:

```bash
conda create --name d2l python=3.8 -y
```


## Downloading the D2L Notebooks

Next, we need to download the code of this book. 
You can click the "All Notebooks" tab 
on the top of any HTML page 
to download and unzip the code.
Alternatively, if you have `unzip` 
(otherwise run `sudo apt install unzip`) available:

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
```


Now we can activate the `d2l` environment:

```bash
conda activate d2l
```


## Installing the Framework and the `d2l` Package

Before installing any deep learning framework, 
please first check whether or not 
you have proper GPUs on your machine
(the GPUs that power the display 
on a standard laptop are not relevant for our purposes).
If you are working on a GPU server,
proceed to :ref:`subsec_gpu` 
for instructions on how 
to install GPU-friendly versions
of the relevant libraries.

If your machine does not house any GPUs,
there is no need to worry just yet.
Your CPU provides more than enough horsepower 
to get you through the first few chapters.
Just remember that you will want to access GPUs 
before running larger models.
To install the the CPU version,
execute the following command.


:begin_tab:`mxnet`

```bash
pip install mxnet==1.7.0.post1
```


:end_tab:


:begin_tab:`pytorch`

```bash
pip install torch torchvision
```


:end_tab:

:begin_tab:`tensorflow`
You can install TensorFlow with both CPU and GPU support as follows:

```bash
pip install tensorflow tensorflow-probability
```


:end_tab:


Our next step is to install 
the `d2l` package that we developed 
in order to encapsulate
frequently used functions and classes
found throughout this book.

```bash
# -U: Upgrade all packages to the newest available version
pip install -U d2l
```


Once you have completed these installation steps, we can the Jupyter notebook server by running:

```bash
jupyter notebook
```


At this point, you can open http://localhost:8888 
(it may have already opened automatically) in your Web browser. 
Then we can run the code for each section of the book.
Please always execute `conda activate d2l` 
to activate the runtime environment
before running the code of the book 
or updating the deep learning framework or the `d2l` package.
To exit the environment, 
run `conda deactivate`.


## GPU Support
:label:`subsec_gpu`

:begin_tab:`mxnet`
By default, MXNet is installed without GPU support
to ensure that it will run on any computer (including most laptops).
Part of this book requires or recommends running with GPU.
If your computer has NVIDIA graphics cards and has installed [CUDA](https://developer.nvidia.com/cuda-downloads),
then you should install a GPU-enabled version.
If you have installed the CPU-only version,
you may need to remove it first by running:

```bash
pip uninstall mxnet
```


We now need to find out what version of CUDA you have installed.
You can check this by running `nvcc --version` 
or `cat /usr/local/cuda/version.txt`.
Assume that you have installed CUDA 10.1,
then you can install with the following command:

```bash
# For Windows users
pip install mxnet-cu101==1.7.0 -f https://dist.mxnet.io/python

# For Linux and macOS users
pip install mxnet-cu101==1.7.0
```


You may change the last digits according to your CUDA version, e.g., `cu100` for
CUDA 10.0 and `cu90` for CUDA 9.0.
:end_tab:


:begin_tab:`pytorch,tensorflow`
By default, the deep learning framework is installed with GPU support.
If your computer has NVIDIA GPUs and has installed [CUDA](https://developer.nvidia.com/cuda-downloads),
then you are all set.
:end_tab:

## Exercises

1. Download the code for the book and install the runtime environment.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/23)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/24)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/436)
:end_tab:
