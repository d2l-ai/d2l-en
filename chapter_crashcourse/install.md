# Getting started with Gluon

This section discusses how to download the codes in this book and to install the software needed to run them. Although skipping this section will not affect your theoretical understanding of sections to come, we strongly recommend that you get some hands-on experience. We believe that  modifying codes and observing their results greatly enhances the benefit you can gain from the book.

For simplicity we recommend [Conda](https://conda.io), a popular Python package manager to install all libraries.

### Windows

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

这时在浏览器打开 http://localhost:8888 （通常会自动打开）就可以查看和运行本书中每一节的代码了。

In some chapters of this book, codes will automatically download the data set and pre-training model; use the US sites to download them by default. We can use domestic sites to download data and models by specifying that MXNet do so prior to running Jupyter.

```
set MXNET_GLUON_REPO=https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/ jupyter notebook
```

### Linux/macOS users

第一步：根据操作系统下载Miniconda（网址：https://conda.io/miniconda.html ），它是一个sh文件。打开Terminal应用进入命令行来执行这个sh文件，例如

```
sh Miniconda3-latest-Linux-x86_64.sh
```

the terms of use will be displayed during installation, press "↓" to continue reading, or press "Q" to exit reading. After that, you will be required to answer the following questions:

```
Do you accept the license terms? [yes|no]
[no] >>> yes
Do you wish the installer to prepend the Miniconda3 install location
to PATH in your /home/your_name/.conda ? [yes|no]
[no] >>> yes
```

After the installation is complete, you need to make Conda take effect. Linux users will need to run `source ~/.bashrc` or restart the command line application; macOS users will need to run `source ~/.bash_profile` or restart the command line application.

Step 2: Download the tarball containing all the codes provided in the book. After unzipping it, run the following commands. Run the following commands.

```
mkdir gluon_tutorials_zh-1.0 && cd gluon_tutorials_zh-1.0
curl https://zh.gluon.ai/gluon_tutorials_zh-1.0.tar.gz -o tutorials.tar.gz
tar -xzvf tutorials.tar.gz && rm tutorials.tar.gz
```

For Steps 3-5, please refer to the installation steps as defined for Windows users.  However, for Step 4, the command should be replaced with

```
source activate gluon
```

## Updating Codes and Runtime Environment

To stay updated on the rapid development of deep learning and MXNet, the open source content in this book will be regularly updated and released. We suggest that you update the open source content (such as codes) and the corresponding runtime environment (such as new versions of MXNet) on a regular basis.  The specific steps for the update are as follows.

Step 1: Re-download the latest tarball containing all the codes provided in this book.  You can download the tarball from one of the following addresses.

* https://zh.gluon.ai/gluon_tutorials_zh.zip
* https://zh.gluon.ai/gluon_tutorials_zh.tar.gz

After unzipping it, visit the folder "gluon_tutorials_zh".

Step 2: Use the following command to update the runtime environment.

```
conda env update -f environment.yml
```

The subsequent steps for activating environment and running Jupyter are the same as those described in the section above.


## Using GPU-enabled version of MXNet

The MXNet to be installed by following the steps described above supports CPU computation only.  Some chapters in this book, however, require or recommend running MXNet via a GPU.   If you have a Nvidia graphics card on your computer and have CUDA installed already, it is recommended that you use the GPU-enabled version of MXNet.

Step 1: Uninstall the CPU-enabled version of MXNet.  If there is no virtual environment installed, you may skip this step.  If you have installed a virtual environment, the runtime environment must be activated prior to uninstallation:

```
pip uninstall mxnet
```

Then, exit the virtual environment by using the command `deactivate` for Windows users, and the command `source deactivate` for Linux/macOS users.

Step 2: Update the environment to the GPU-dependent MXNet. Open the file `environment.yml` in the root directory where the codes in this book are contained, and with a text editor, replace "mxnet" located within the file with the corresponding GPU-version. For example, if the 8.0 version of CUDA is installed on the computer, change the string "mxnet" located within the file to "mxnet-cu80".  If other versions of CUDA (such as versions 7.5, 9.0, 9.2, etc.) are installed, complete the appropriate changes to the string "mxnet" located within the file (for example, change the line to "mxnet-cu75", "mxnet-cu90" or "mxnet-cu92", etc.). Save and exit the file.

Step 3: Update the virtual environment and execute the command.

```
conda env update -f environment.yml
```

Afterward, we only need to activate the installation environment to use the GPU-enabled version of MXNet to run the codes provided in the book. One thing to keep in mind: if new codes are downloaded later, you will need to repeat these three steps to use the GPU-enabled version of MXNet.


## Summary

* To get a hands-on learning experience with deep learning, we need to acquire the codes provided in this book and install the appropriate runtime environment.
* We recommend that you update the codes and runtime environment.


## exercise

* Acquire the codes in this book and install the runtime environment. If you encounter any problem during installation, please scan the QR code in this section. On the forum, you can see our FAQ or ask your own questions.

## Scan the QR code to get to the [forum](https://discuss.gluon.ai/t/topic/249)

![](../img/qr_install.svg)
