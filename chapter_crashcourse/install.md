# Getting started with Gluon

To get started we need to download and install the code needed to run the notebooks. Although skipping this section will not affect your theoretical understanding of sections to come, we strongly recommend that you get some hands-on experience. We believe that modifying and writing code and seeing the results thereof greatly enhances the benefit you can gain from the book. In a nutshell, to get started you need to do the following steps:

1. Install Conda
1. Download the code that goes with the book
1. Install GPU drivers if you have a GPU and haven't used it before
1. Build the Conda environment to run MXNet and the examples of the book


## Conda

For simplicity we recommend [Conda](https://conda.io), a popular Python package manager to install all libraries.

1. Download and install [Miniconda](https://conda.io/miniconda.html) at [conda.io/miniconda.html](https://conda.io/miniconda.html) based on your operating system. 
1. Update your shell by `source ~/.bashrc` (Linux) or `source ~/.bash_profile` (macOS). Make sure to add Anaconda to your PATH environment variable.
1. Download the tarball containing the notebooks from this book. This can be found at (www.diveintodeeplearning.org/d2l-en-1.0.zip)[https://www.diveintodeeplearning.org/d2l-en-1.0.zip]. Alternatively feel free to clone the latest version from GitHub.
1. Uncompress the ZIP file and move its contents to a folder for the tutorials.

On Linux this can be accomplished as follows from the command line; For MacOS replace Linux by MacOSX in the first line, for Windows follow the links provided above. 

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
mkdir d2l-en
cd d2l-en
curl https://www.diveintodeeplearning.org/d2l-en-1.0.zip -o d2l-en.zip
unzip d2l-en-1.0.zip
rm d2l-en-1.0.zip
```

## GPU Support

By default MXNet is installed without GPU support to ensure that it will run on any computer (including most laptops). If you should be so lucky to have a GPU enabled computer, you should modify the Conda environment to download the CUDA enabled build. Obviously you need to have the appropriate drivers installed. In particular you need the following:

1. Ensure that you install the [NVIDIA Drivers](https://www.nvidia.com/drivers) for your specific GPU. 
1. Install [CUDA](https://developer.nvidia.com/cuda-downloads), the programming language for GPUs.
1. Install [CUDNN](https://developer.nvidia.com/cudnn), which contains many optimized libraries for deep learning.
1. Install [TensorRT](https://developer.nvidia.com/tensorrt), if appropriate, for further acceleration. 

The installation process is somewhat lengthy and you will need to agree to a number of different licenses and use different installation scripts for it. Details will depend strongly on your choice of operating system and hardware. 

Next update the environment description in `environment.yml`. Replace `mxnet` by `mxnet-cu92` or whatever version of CUDA that you've got installed. For instance, if you're on CUDA 8.0, you need to replace `mxnet-cu92` with `mxnet-cu80`. You should do this *before* creating the Conda environment. Otherwise you will need to rebuild it later. On Linux this looks as follows (on Windows you can use e.g. Notepad to edit `environment.yml` directly).

```
cd d2l
emacs environment.yml
```

## Conda Environment

In a nutshell, Conda provides a mechanism for setting up a set of Python libraries in a reproducible and reliable manner, ensuring that all software dependencies are satisfied. Here's what is needed to get started.

1. Create and activate the environment using Conda. For convenience we created an `environment.yml` file to hold all configuration.
1. Activate the environment.
1. Open Jupyter notebooks to start experimenting.

### Windows

As before, open the command line terminal. 

```
conda env create -f environment.yml
cd d2l-en
activate gluon
jupyter notebook
```

If you need to reactivate the set of libraries later, just skip the first line. This will ensure that your setup is active. Note that instead of Jupyter Notebooks you can also use JupyterLab via `jupyter lab` instead of `jupyter notebook`. This will give you a more powerful Jupyter environment (if you have JupyterLab installed). You can do this manually via `conda install jupyterlab` from within an active conda gluon environment. 

If your browser integration is working properly, starting Jupyter will open a new window in your browser. If this doesn't happen, go to http://localhost:8888 to open it manually. Some notebooks will automatically download the data set and pre-training model. You can adjust the location of the repository by overriding the `MXNET_GLUON_REPO` variable.

### Linux and MacOSX

The steps for Linux are quite similar, just that anaconda uses slightly different command line options. 

```
conda env create -f environment.yml
cd d2l-en
source activate gluon
jupyter notebook
```

The main difference between Windows and other installations is that for the former you use `activate gluon` whereas for Linux and macOS you use `source activate gluon`. Beyond that, the same considerations as for Windows apply. Install JupyterLab if you need a more powerful environment 

## Updating Gluon

In case you want to update the repository, if you installed a new version of CUDA and (or) MXNet, you can simply use the Conda commands to do this. As before, make sure you update the packages accordingly.

```
cd d2l-en
conda env update -f environment.yml
```

## Summary

* Conda is a Python package manager that ensures that all software dependencies are met.
* `environment.yml` has the full configuration for the book. All notebooks are available for download or on GitHub.
* Install GPU drivers and update the configuration if you have GPUs. This will shorten the time to train significantly. 

## Exercise

1. Download the code for the book and install the runtime environment. 
1. Follow the links at the bottom of the section to the forum in case you have questions and need further help.
1. Create an account on the forum and introduce yourself.

## Discuss on our Forum

<div id="discuss" topic_id="2315"></div>
