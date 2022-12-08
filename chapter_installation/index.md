# Installation
:label:`chap_installation`

In order to get up and running,
we will need an environment for running Python,
the Jupyter Notebook, the relevant libraries,
and the code needed to run the book itself.

## Installing Miniconda

Your simplest option is to install
[Miniconda](https://conda.io/en/latest/miniconda.html).
Note that the Python 3.x version is required.
You can skip the following steps
if your machine already has conda installed.

Visit the Miniconda website and determine
the appropriate version for your system
based on your Python 3.x version and machine architecture.
Suppose that your Python version is 3.9
(our tested version).
If you are using macOS,
you would download the bash script
whose name contains the strings "MacOSX",
navigate to the download location,
and execute the installation as follows
(taking Intel Macs as an example):

```bash
# The file name is subject to changes
sh Miniconda3-py39_4.12.0-MacOSX-x86_64.sh -b
```


A Linux user
would download the file
whose name contains the strings "Linux"
and execute the following at the download location:

```bash
# The file name is subject to changes
sh Miniconda3-py39_4.12.0-Linux-x86_64.sh -b
```


Next, initialize the shell so we can run `conda` directly.

```bash
~/miniconda3/bin/conda init
```


Then close and reopen your current shell.
You should be able to create
a new environment as follows:

```bash
conda create --name d2l python=3.9 -y
```


Now we can activate the `d2l` environment:

```bash
conda activate d2l
```


## Installing the Deep Learning Framework and the `d2l` Package

Before installing any deep learning framework,
please first check whether or not
you have proper GPUs on your machine
(the GPUs that power the display
on a standard laptop are not relevant for our purposes).
For example,
if your computer has NVIDIA GPUs and has installed [CUDA](https://developer.nvidia.com/cuda-downloads),
then you are all set.
If your machine does not house any GPU,
there is no need to worry just yet.
Your CPU provides more than enough horsepower
to get you through the first few chapters.
Just remember that you will want to access GPUs
before running larger models.


:begin_tab:`mxnet`

To install a GPU-enabled version of MXNet,
we need to find out what version of CUDA you have installed.
You can check this by running `nvcc --version`
or `cat /usr/local/cuda/version.txt`.
Assume that you have installed CUDA 10.2,
then execute the following command:

```bash
# For macOS and Linux users
pip install mxnet-cu102==1.7.0

# For Windows users
pip install mxnet-cu102==1.7.0 -f https://dist.mxnet.io/python
```


You may change the last digits according to your CUDA version, e.g., `cu101` for
CUDA 10.1 and `cu90` for CUDA 9.0.


If your machine has no NVIDIA GPUs
or CUDA,
you can install the CPU version
as follows:

```bash
pip install mxnet==1.7.0.post1
```


:end_tab:


:begin_tab:`pytorch`

You can install PyTorch with either CPU or GPU support as follows:

```bash
pip install torch torchvision
```


:end_tab:

:begin_tab:`tensorflow`
You can install TensorFlow with either CPU or GPU support as follows:

```bash
pip install tensorflow tensorflow-probability
```


:end_tab:


Our next step is to install
the `d2l` package that we developed
in order to encapsulate
frequently used functions and classes
found throughout this book:

```bash
pip install d2l==1.0.0a1.post0
```


## Downloading and Running the Code

Next, you will want to download the notebooks
so that you can run each of the book's code blocks.
Simply click on the "Notebooks" tab at the top
of any HTML page on [the D2L.ai website](https://d2l.ai/)
to download the code and then unzip it.
Alternatively, you can fetch the notebooks
from the command line as follows:

:begin_tab:`mxnet`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd mxnet
```


:end_tab:


:begin_tab:`pytorch`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd pytorch
```


:end_tab:

:begin_tab:`tensorflow`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd tensorflow
```


:end_tab:

If you don't already have `unzip` installed, first run `sudo apt-get install unzip`.
Now we can start the Jupyter Notebook server by running:

```bash
jupyter notebook
```


At this point, you can open http://localhost:8888
(it may have already opened automatically) in your Web browser.
Then we can run the code for each section of the book.
Whenever you open a new command line window,
you will need to execute `conda activate d2l`
to activate the runtime environment
before running the D2L notebooks,
or updating your packages
(either the deep learning framework
or the `d2l` package).
To exit the environment,
run `conda deactivate`.



:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/23)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/24)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/436)
:end_tab:
