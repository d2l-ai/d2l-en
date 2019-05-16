# Installation
:label:`chapter_installation`

To get you up and running with hands-on experiences, we'll need you to set up with a Python environment, Jupyter's interactive notebooks, the relevant libraries, and the code needed to *run the book*.


## Obtaining Code and Installing Running Environment

This book with code can be downloaded for free. For simplicity we recommend conda, a popular Python package manager to install all libraries. Windows users and Linux/macOS users can follow the instructions below respectively.

### Windows Users

If it is your first time to run the code of this book, you need to complete the following 5 steps. Next time you can jump directly to Step 4 and Step 5.

Step 1 is to download and install [Miniconda](https://conda.io/en/master/miniconda.html) according to the operating system in use. During the installation, it is required to choose the option "Add Anaconda to the system PATH environment variable".

Step 2 is to download the compressed file containing the code of this book. It is available at https://www.d2l.ai/d2l-en.zip. After downloading the zip file, create a folder `d2l-en` and extract the zip file into the folder. At the current folder, enter `cmd` in the address bar of File Explorer to enter the command line mode.

Step 3 is to create a virtual (running) environment using conda to install the libraries needed by this book. Here `environment.yml` is placed in the downloaded zip file. Open the file with a text editor to see the libraries (such as MXNet and `d2lzh` package) and their version numbers on which running the code of the book is dependent.

```
conda env create -f environment.yml
```

Step 4 is to activate the environment that is created earlier. Activating this environment is a prerequisite for running the code of this book. To exit the environment, use the command `conda deactivate` (if the conda version is lower than 4.4, use the command `deactivate`).

```
# If the conda version is lower than 4.4, use the command `activate gluon`
conda activate gluon
```

Step 5 is to open the Jupyter Notebook.

```
jupyter notebook
```

At this point open http://localhost:8888 (which usually opens automatically) in the browser, then you can view and run the code in each section of the book.

### Linux/macOS Users

Step 1 is to download and install [Miniconda](https://conda.io/en/master/miniconda.html) according to the operating system in use. It is a sh file. Open the Terminal application and enter the command to execute the sh file, such as

```
# The file name is subject to change, always use the one downloaded from the
# Miniconda website
sh Miniconda3-latest-Linux-x86_64.sh
```

The terms of use will be displayed during installation. Press "↓" to continue reading, press "Q" to exit reading. After that, answer the following questions:

```
Do you accept the license terms? [yes|no]
[no] >>> yes
Do you wish the installer to prepend the Miniconda3 install location
to PATH in your /home/your_name/your_file ? [yes|no]
[no] >>> yes
```

After the installation is complete, conda should be made to take effect. Linux users need to run `source ~/.bashrc` or restart the command line application; macOS users need to run `source ~/.bash_profile` or restart the command line application.

Step 2 is to create a folder `d2l-en`, download the compressed file containing the code of this book, and extract it into the folder. Linux users who have not already installed `unzip` can run the command `sudo apt-get install unzip` to install it. Run the following commands:

```
mkdir d2l-en && cd d2l-en
curl https://www.d2l.ai/d2l-en.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
```

For Step 3 to Step 5, refer to the steps for Windows users as described earlier. If the conda version is lower than 4.4, replace the command in Step 4 with `source activate gluon` and exit the virtual environment using the command `source deactivate`.


## Updating Code and Running Environment

Since deep learning and MXNet grow fast, this open source book will be updated and released regularly. To update the open source content of this book (e.g., code) with a corresponding running environment (e.g., MXNet of a later version), follow the steps below.

Step 1 is to re-download the latest compressed file containing the code of this book. It is available at https://www.d2l.ai/d2l-en.zip. After extracting the zip file, enter the folder `d2l-en`.

Step 2 is to update the running environment with the command

```
conda env update -f environment.yml
```

The subsequent steps for activating the environment and running Jupyter are the same as those described earlier.

## GPU Support

By default MXNet is installed without GPU support to ensure that it will run on any computer (including most laptops). Part of this book requires or recommends running with GPU. If your computer has NVIDIA graphics cards and has installed CUDA, you should modify the conda environment to download the CUDA enabled build.

Step 1 is to uninstall MXNet without GPU support. If you have installed the virtual environment for running the book, you need to activate this environment and then uninstall MXNet without GPU support:

```
pip uninstall mxnet
```

Then exit the virtual environment.

Step 2 is to update the environment description in `environment.yml`. 
Likely, you'll want to replace `mxnet` by `mxnet-cu90`.
The number following the hyphen (90 above) corresponds to the version of CUDA you installed). For instance, if you're on CUDA 8.0, you need to replace `mxnet-cu90` with `mxnet-cu80`. You should do this *before* creating the conda environment. Otherwise you will need to rebuild it later.

Step 3 is to update the virtual environment. Run the command

```
conda env update -f environment.yml
```

Now we only need to activate the virtual environment to use MXNet with GPU support to run the book. Note that you need to repeat these 3 steps to use MXNet with GPU support if you download the updated code later.

## Exercises

1. Download the code for the book and install the runtime environment.


## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2315)

![](../img/qr_install.svg)
