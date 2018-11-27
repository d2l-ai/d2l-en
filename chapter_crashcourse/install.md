# Getting started with Gluon

This section discusses how to download the codes in this book and to install the software needed to run them. Although skipping this section will not affect your theoretical understanding of sections to come, we strongly recommend that you get some hands-on experience. We believe that  modifying codes and observing their results greatly enhances the benefit you can gain from the book.

## Conda

For simplicity we recommend [Conda](https://conda.io), a popular Python package manager to install all libraries.

1. Download and install [Miniconda](https://conda.io/miniconda.html), based on your operating system. Make sure to add Anaconda to your PATH environment variable.
1. Download the tarball containing the notebooks from this book. This can be found at en.gluon.ai/gluon_tutorials_en-latest.zip. Alternatively feel free to clone the latest version from GitHub.
1. Unzip the the tarball and move its contents to a folder for the tutorials.

### GPU Support

By default MXNet is installed without GPU support to ensure that it will run on any computer (including most laptops). If you should be so lucky to have a GPU enabled computer, you should modify the Conda environment to download the CUDA enabled build. Obviously you need to have the appropriate drivers installed, such as [CUDA](https://developer.nvidia.com/cuda-downloads), [CUDNN](https://developer.nvidia.com/cudnn) and [TensorRT](https://developer.nvidia.com/tensorrt), if appropriate.

Next update the environment description in `environment.yml`. Replace `mxnet` by `mxnet-cu92` or whatever version of CUDA that you've got installed. For instance, if you're on CUDA 8.0, you need to replace `mxnet-cu92` with `mxnet-cu80`. You should do this *before* creating the Conda environment. Otherwise you will need to rebuild it later.


### Windows

1. Create and activate the environment using Conda. For convenience we created an `environment.yml` file to hold all configuration.
1. Activate the environment.
1. Open Jupyter notebooks to start experimenting.

```
conda env create -f environment.yml
activate gluon
jupyter notebook
```

Alternatively, you can open `jupyter-lab` instead of `jupyter notebook`. This will give you a more powerful Jupyter environment.
If your browser integration is working properly, starting Jupyter will open a new window in your browser. If this doesn't happen, go to http://localhost:8888 to open it manually.

Some notebooks will automatically download the data set and pre-training model. You can adjust the location of the repository by overriding the `MXNET_GLUON_REPO` variable.

### Linux and macOS

Installation for both is very similar. We give a description of the workflow for Linux.

1. Install Miniconda (and accepting the license terms), as available at https://conda.io/miniconda.html
1. Update your shell by `source ~/.bashrc` (Linux) or `source ~/.bash_profile` (macOS) or open a new terminal.
1. Download the tar file with all code and unpack it.
1. Create the Conda environment
1. Activate it and start Jupyter

```
sh Miniconda3-latest-Linux-x86_64.sh

mkdir gluon_tutorials
cd gluon_tutorials
curl https://en.gluon.ai/gluon_tutorials_en-latest.tar.gz -o tutorials.tar.gz
tar -xzvf tutorials.tar.gz
rm tutorials.tar.gz

conda env create -f environment.yml
source activate gluon
jupyter notebook
```

The main difference between Windows and other installations is that for the former you use `activate gluon` whereas for Linux and macOS you use `source activate gluon`.

```
conda env update -f environment.yml
activate gluon
```

## Updating Gluon

In case you want to update the repository, if you installed a new version of CUDA and (or) MXNet, you can simply use the Conda commands to do this. As before, make sure you update the packages accordingly.

```
conda env update -f environment.yml
```

## Exercise

Download the code for the book and install the runtime environment. If you encounter any problems during installation, please scan the QR code to take you to the FAQ section of the discussion forum for further help.

## Discuss on our Forum

<div id="discuss" topic_id="2315"></div>
