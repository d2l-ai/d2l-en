# Installing MXNet and Gluon

To get you up and running with hands-on experience
we'll need you to set up with a Python environment,
jupyter's interactive notebooks,
the relevant libraries, and the code need to *run the book*.
In this section, we will guide you through the following steps:

1. Install conda
1. Download the code that goes with the book
1. Install GPU drivers if you have a GPU and haven't used it before
1. Build the conda environment to run MXNet and the examples of the book


## Installing Conda

For simplicity we recommend [conda](https://conda.io), a popular Python package manager to install all libraries.

1. Download and install [Miniconda](https://conda.io/miniconda.html) at [conda.io/miniconda.html](https://conda.io/miniconda.html) based on your operating system.
1. Update your shell by `source ~/.bashrc` (Linux) or `source ~/.bash_profile` (macOS). Make sure to add Anaconda to your PATH environment variable.
1. Download the tarball containing the notebooks from this book. This can be found at [www.d2l.ai/d2l-en-1.0.zip](https://www.d2l.ai/d2l-en-1.0.zip). Alternatively feel free to clone the latest version from GitHub.
1. Uncompress the ZIP file and move its contents to a folder for the tutorials.

On Linux this can be accomplished as follows from the command line; For MacOS replace Linux by MacOSX in the first line, for Windows follow the links provided above.

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
mkdir d2l-en
cd d2l-en
curl https://www.d2l.ai/d2l-en-1.0.zip -o d2l-en.zip
unzip d2l-en-1.0.zip
rm d2l-en-1.0.zip
```

## GPU Support

By default MXNet is installed without GPU support
to ensure that it will run on any computer (including most laptops).
If you're fortunate enough to be running on a GPU-enabled computer,
you should modify the conda environment to download the CUDA enabled build.
Not that you will need to have the appropriate drivers installed.
In particular you need to:

1. Ensure that you install the [NVIDIA Drivers](https://www.nvidia.com/drivers)
for your specific GPU.
1. Install [CUDA](https://developer.nvidia.com/cuda-downloads),
the programming language for GPUs.
1. Install [CUDNN](https://developer.nvidia.com/cudnn),
which contains many optimized libraries for deep learning.
1. Install [TensorRT](https://developer.nvidia.com/tensorrt),
if appropriate, for further acceleration.

The installation process is somewhat lengthy
and you will need to agree to a number of different licenses.
Details will depend strongly on your choice of operating system and hardware.

Next, update the environment description in `environment.yml`.
Likely, you'll want to replace `mxnet` by `mxnet-cu92`.
The number following the hyphen (92 above)
corresponds to the version of CUDA you installed).
For instance, if you're on CUDA 8.0,
you need to replace `mxnet-cu92` with `mxnet-cu80`.
You should do this *before* creating the conda environment.
Otherwise you will need to rebuild it later.
On Linux this looks as follows
(on Windows you can use e.g. Notepad to edit `environment.yml` directly).

```
cd d2l
emacs environment.yml
```

## Conda Environment

Conda provides a mechanism for managing Python libraries
in a reproducible and reliable manner.
To set up your environment, you'll need to:

1. Create and activate the environment using conda. For convenience we created an `environment.yml` file to hold all configuration.
1. Activate the environment.
1. Open Jupyter notebooks to start experimenting.

## Windows

As before, open the command line terminal.

```
conda env create -f environment.yml
cd d2l-en
activate gluon
jupyter notebook
```

If you need to reactivate the set of libraries later, just skip the first line.
This will ensure that your setup is active.
Note that instead of Jupyter Notebooks you can also use JupyterLab
via `jupyter lab` instead of `jupyter notebook`.
This will give you a more powerful Jupyter environment
(if you have JupyterLab installed).
You can do this manually via `conda install jupyterlab`
from within an active conda gluon environment.

If your browser integration is working properly,
starting Jupyter will open a new window in your browser.
If this doesn't happen, go to http://localhost:8888 to open it manually.
Some notebooks will automatically download the data set and pre-training model.
You can adjust the location of the repository by overriding the `MXNET_GLUON_REPO` variable.

## Linux and MacOSX

The steps for Linux are similar but instead of `activate gluon`, you'll want to run `source activate gluon`.

```
conda env create -f environment.yml
cd d2l-en
source activate gluon
jupyter notebook
```


## Updating Gluon

In case you want to update the repository,
if you installed a new version of CUDA and (or) MXNet,
you can simply use the conda commands to do this.
As before, make sure you update the packages accordingly.

```
cd d2l-en
conda env update -f environment.yml
```

## Exercises
1. Download the code for the book and install the runtime environment.


## Discuss on our Forum


<div id="discuss" topic_id="2315"></div>

