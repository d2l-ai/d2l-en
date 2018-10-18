# Obtaining and Running Codes in This Book

This section will describe how one can attain the codes provided by this book and install the software needed to run them. Although skipping this section will not affect your understanding of sections to come, we strongly recommend that you follow the steps outlined below to practice our provided hands-on activities. Most exercises in this book are related to modifying codes and observing their results. Therefore, the successful completion of these exercises is solely based on the knowledge acquired in this section.

## Acquiring Codes and Installing Runtime Environment

The content of this book and the codes provided in it are available online for free. We recommend using Conda to install the dependent software that runs the codes. Conda is a popular Python package management software. For Windows and Linux/macOS users, please refer to the corresponding content below.

### Windows users

When first running the program, it will require the completion of the following five steps. After your initial running of the program, you may then ignore the previous three steps regarding its download and installation and skip directly to Steps 4 and 5.

第一步：根据操作系统下载并安装Miniconda（网址：https://conda.io/miniconda.html ），在安装过程中需要勾选“Add Anaconda to my PATH environment variable”选项。

Step 2: Download the tarball containing all the codes in this book. This file can be downloaded by entering the following address in the address bar of your preferred browser and pressing Enter:

> https://zh.gluon.ai/gluon_tutorials_zh-1.0.zip

Once the download is complete, create a folder named "gluon_tutorials_zh-1.0" and unzip the tarball discussed above to this folder. Enter `cmd` found in the address bar of the directory file explorer into the command line mode.

Step 3: Create and activate the environment using Conda. By default, Conda downloads software from international sites. The following optional configurations can be used to speed up downloads from Chinese mirror sites:

```
# From Tsinghua Conda mirror site.
conda config --prepend channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/

# Or from USTC Conda mirror site.
conda config --prepend channels http://mirrors.ustc.edu.cn/anaconda/pkgs/free/
```

Next, will use Conda to create a virtual environment and install the software needed for this book. `environment.yml` here is a file placed in the code tarball that specifies what software is needed to execute the codes provided in the book.

```
conda env create -f environment.yml
```

Step 4: Activate the environment you created earlier.

```
activate gluon
```

Step 5: Open the Jupyter notebook.

```
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
