# Installation
:label:`chap_installation`

In order to get you up and running for hands-on learning experience, 
we need to set up you up with an environment for running Python, 
Jupyter notebooks, the relevant libraries, 
and the code needed to run the book itself.

## Installing Miniconda

The simplest way to get going will be to install [Miniconda](https://conda.io/en/latest/miniconda.html). 
Download the corresponding Miniconda "sh" file from the website 
and then execute the installation from the command line
using `sudo sh <FILENAME>` as follows:

```bash
# For Mac users (the file name is subject to changes)
sudo sh Miniconda3-latest-MacOSX-x86_64.sh

# For Linux users (the file name is subject to changes)
sudo sh Miniconda3-latest-Linux-x86_64.sh
```


You will be prompted to answer the following questions:

```bash
Do you accept the license terms? [yes|no]
[no] >>> yes

Miniconda3 will now be installed into this location:
/home/rlhu/miniconda3
  - Press ENTER to confirm the location
  - Press CTRL-C to abort the installation
  - Or specify a different location below
>>> <ENTER>

Do you wish the installer to initialize Miniconda3
by running conda init? [yes|no]
[no] >>> yes
```


After installing miniconda, run the appropriate command 
(depending on your operating system) to activate conda.

```bash
# For Mac user
source ~/.bash_profile

# For Linux user
source ~/.bashrc
```


Then create the conda "d2l"" environment and enter `y`
for the following inquiries as shown in :numref:`fig_conda_create_d2l`.

```bash
conda create --name d2l
```


![ Conda create environment d2l. ](../img/conda_create_d2l.png)
:width:`700px`
:label:`fig_conda_create_d2l`


## Downloading the d2l Notebooks

Next, we need to download the code for this book.

```bash
sudo apt-get install unzip
mkdir d2l-en && cd d2l-en
wget http://d2l.ai/d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
```

Now we will now want to activate the "d2l" environment and install `pip`. 
Enter `y` for the queries that follow this command.

```bash
conda activate d2l
conda install python=3.7 pip
```


Finally, install the "d2l" package within the environment "d2l" that we created.

```
pip install git+https://github.com/d2l-ai/d2l-en@numpy2
```


If everything went well up to now then you are almost there.
If something went wrong, please check the following:

1. That you are using `pip` for Python 3 instead of Python 2 by checking `pip --version`. If it is Python 2, then you may check if there is a `pip3` available.
2. That you are using a recent `pip`, such as version 19. 
   If not, you can upgrade it via `pip install --upgrade pip`.
3. Whether you have permission to install system-wide packages. 
   If not, you can install to your home directory by adding the flag `--user` 
   to the pip command, e.g. `pip install d2l --user`.


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

```
# For Windows users
pip install mxnet==1.6.0b20190926

# For Linux and macOS users
pip install mxnet==1.6.0b20190915
```


Once both packages are installed, we now open the Jupyter notebook by running:

```
jupyter notebook
```


At this point, you can open http://localhost:8888 (it usually opens automatically) in your web browser. Once in the notebook server, we can run the code for each section of the book.

## Upgrade to a New Version

Both this book and MXNet are keeping improving. Please check a new version from time to time.

1. The URL  http://numpy.d2l.ai/d2l-en.zip always points to the latest contents.
2. Please upgrade "d2l" by `pip install git+https://github.com/d2l-ai/d2l-en@numpy2`.
3. For the CPU version, MXNet can be upgraded by `pip uninstall mxnet` then re-running the aforementioned `pip install mxnet==...` command.


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
then you can install the according MXNet version 
with the following (OS-specific) command:

```
# For Windows users
pip install mxnet-cu101==1.6.0b20190926

# For Linux and macOS users
pip install mxnet-cu101==1.6.0b20190915
```


You may change the last digits according to your CUDA version,
e.g., `cu100` for CUDA 10.0 and `cu90` for CUDA 9.0.
You can find all available MXNet versions via `pip search mxnet`.

For installation of MXNet on other platforms, please refer to http://numpy.mxnet.io/#installation.


## Exercises

1. Download the code for the book and install the runtime environment.


## [Discussions](https://discuss.mxnet.io/t/2315)

![](../img/qr_install.svg)
