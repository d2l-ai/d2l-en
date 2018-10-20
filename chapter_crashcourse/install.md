# Getting started with Gluon

This section discusses how to download the codes in this book and to install the software needed to run them. Although skipping this section will not affect your theoretical understanding of sections to come, we strongly recommend that you get some hands-on experience. We believe that  modifying codes and observing their results greatly enhances the benefit you can gain from the book.

For simplicity we recommend [Conda](https://conda.io), a popular Python package manager to install all libraries.

## Windows

1. Download and install [Miniconda](https://conda.io/miniconda.html), based on your operating system. Make sure to add Anaconda to your PATH environment variable.
1. Download the tarball containing the notebooks from this book. This can be found at en.gluon.ai/gluon_tutorials_en-latest.zip. Alternatively feel free to clone the latest version from GitHub.
1. Unzip the the tarball and move its contents to a folder for the tutorials.
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

## Linux and macOS

Installation for both is very similar. We give a description of the workflow for Linux. Begin by installing Miniconda (and accepting the liense terms), as available at https://conda.io/miniconda.html via  

```
sh Miniconda3-latest-Linux-x86_64.sh
```

Next you need to update your shell for Conda to take effect. Linux users will need to run `source ~/.bashrc` or restart the command line application; macOS users will need to run `source ~/.bash_profile` or restart the command line application.

Download the tarball containing all the codes provided in the book and unpack it.

```
mkdir gluon_tutorials
cd gluon_tutorials
curl https://en.gluon.ai/gluon_tutorials_en-latest.tar.gz -o tutorials.tar.gz
tar -xzvf tutorials.tar.gz
rm tutorials.tar.gz
```

Next (in analogy to Windows) create the appropriate Conda environment and start Jupyter.

```
conda env create -f environment.yml
source activate gluon
jupyter notebook
```

## Updating Codes and Runtime Environment

We recommend that you download the latest version of the library regularly to keep track of the progress. The code can always be found at `https://en.gluon.ai/gluon_tutorials_en-latest.tar.gz`.

To update the conda environment use the following steps:

```
conda env update -f environment.yml
source activate gluon
```
On Windows you need to replace this as follows:

```
conda env update -f environment.yml
activate gluon
```

## Using the GPU-enabled version of MXNet

The version of MXNet described above supports CPU only. Some chapters in this book require GPUs for efficient experiments (running the code on CPUs would simply take too long). If you have na NVIDIA graphics card on your computer and have CUDA installed already, it is recommended that you use the GPU-enabled version instead.

Uninstall the CPU version of MXNet. Note that if you have a virtual environment installed, you must activate this prior to uninstalling MXNet. On Windows use

```
activate gluon
pip uninstall mxnet
deactivate
```

On Linux and macOS use
```
source activate gluon
pip uninstall mxnet
source deactivate
```

Next update the environment description in `environment.yml` by replacing `mxnet` by `mxnet-cu92`. If you have an oder version of cuda installed, e.g. 8.0 or 9.0, you need to replace this with mxnet-cu80 and mxnet-cu90 respectively.

Finally, update the virtual environment and execute the command.

```
conda env update -f environment.yml
```

Finally, you only need to activate the environment via `source activate gluon` or `activate gluon` for Linux/macOS and Windows respectively. If you download new code or update your version of CUDA, you obviously need to update the entries in `environment.yml` accordingly.

## Exercise

Download the code for the book and install the runtime environment. If you encounter any problems during installation, please scan the QR code to take you to the FAQ section of the discussion forum for further help.

## Scan the QR code to get to the [forum](https://discuss.gluon.ai/t/topic/249)

![](../img/qr_install.svg)
