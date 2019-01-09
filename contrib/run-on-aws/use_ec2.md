# Run on an EC2 instance 
This chapter shows, how to allocate a CPU/GPU instance in AWS and how to setup the Deep Learning environment.

We first need [an AWS account](https://aws.amazon.com/), and then go the EC2 console
after login in.

<img src="https://raw.githubusercontent.com/NRauschmayr/d2l-en/aws_updated/img/aws.png" width="400"/>

Then click "launch instance" to select the operation system and instance type.

<img src="https://raw.githubusercontent.com/NRauschmayr/d2l-en/aws_updated/img/ec2.png" width="500"/>

AWS offers [Deep Learning AMIs](https://docs.aws.amazon.com/dlami/latest/devguide/options.html) 
that come with the latest versions of Deep Learning frameworks. The Deep Learning AMIs provide 
all necessary packages and drivers and allow you to directly start implementing 
and training your models. Deep Learning AMIs use use binaries that are optimized to run on AWS instances to accelerate model training and inference.
In this tutorial we use Deep Learning AMI (Ubuntu) Version 19.0:


<img src="https://raw.githubusercontent.com/NRauschmayr/d2l-en/aws_updated/img/os.png" width="600"/>


We choose "p2.xlarge", which contains a single Nvidia K80 GPU. Note that there is a
large number of instance, refer to
[ec2instances.info](http://www.ec2instances.info/) for detailed configurations
and fees.

<img src="https://raw.githubusercontent.com/NRauschmayr/d2l-en/aws_updated/img/p2x.png" width="400"/>

Note that we need to check the instance limits to guarantee that we can request
the resource. If running out of limits, we can request more capacity by clicking
the right link, which often takes about a single workday to process.

<img src="https://raw.githubusercontent.com/NRauschmayr/d2l-en/aws_updated/img/limits.png" width="500"/>

On the next step we increased the disk from 8 GB to 40 GB so we have enough
space store a reasonable size dataset. For large-scale
datasets, we can "add new volume". Also you selected a very powerful GPU
instance such as "p3.8xlarge", make sure you selected "Provisioned IOPS" in the
volume type for better I/O performance.

<img src="https://raw.githubusercontent.com/NRauschmayr/d2l-en/aws_updated/img/disk.png" width="500"/>

Then we launched with other options as the default values. The last step before
launching is choosing the ssh key, you may need to generate and store a key if
you don't have one before.

<img src="https://raw.githubusercontent.com/NRauschmayr/d2l-en/aws_updated/img/keypair.png" width="400"/>

After clicked "launch instances", we can check the status by clicking the
instance ID link.

<img src="https://raw.githubusercontent.com/NRauschmayr/d2l-en/aws_updated/img/launching.png" width="500"/>

Once the status is green, we can right-click and select "connect" to get the access instruction.

<img src="https://raw.githubusercontent.com/NRauschmayr/d2l-en/aws_updated/img/connect.png" width="500"/>

With the given address, we can log into our instance:

<img src="https://raw.githubusercontent.com/NRauschmayr/d2l-en/aws_updated/img/login_screen.png" width="700"/>

The login screen will show a long list of available conda environments for the different Deep Learning frameworks, CUDA driver and Python versions. With ```conda activate``` you can easily switch into the different environments. In the following example we switch to the MXNet Python 3.6 environment:

<img src="https://raw.githubusercontent.com/NRauschmayr/d2l-en/aws_updated/img/mxnet.png" width="400"/> 

Now you are ready to start developing and training MXNet models. Once you start training, you can check the GPU status with ```nividia-smi```.

## Acquire the Code for this Book and activate MXNet GPU environment

Next, download the code for this book and install and activate the Conda environment.

```
mkdir d2l-en && cd d2l-en
curl https://www.diveintodeeplearning.org/d2l-en-1.0.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
source activate mxnet_p36 
```

## Run Jupyter Notebook

Now, you can run Jupyter Notebook:

```
jupyter notebook
```

Figure 11.18 shows the possible output after you run Jupyter Notebook. The last row is the URL for port 8888.

![ Output after running Jupyter Notebook. The last row is the URL for port 8888. ](../img/jupyter.png)

Because the instance you created does not expose port 8888, you can launch SSH in the local command line and map the instance to the local port 8889.
 
```
# This command must be run in the local command line.
ssh -i "/path/to/key.pem" ubuntu@ec2-xx-xxx-xxx-xxx.y.compute.amazonaws.com -L 8889:localhost:8888
```

Finally, copy the URL shown in the last line of the Jupyter Notebook output in Figure 11.18 to your local browser and change 8888 to 8889. Press Enter to use Jupyter Notebook to run the instance code from your local browser.

## Close Unused Instances

As cloud services are billed by use duration, you will generally want to close instances you no longer use.

If you plan on restarting the instance after a short time, right-click on the example shown in Figure 11.16 and select "Instance State" $\rightarrow$ "Stop" to stop the instance. When you want to use it again, select "Instance State" $\rightarrow$ "Start" to restart the instance. In this situation, the restarted instance will retain the information stored on its hard disk before it was stopped (for example, you do not have to reinstall CUDA and other runtime environments). However, stopped instances will still be billed a small amount for the hard disk space retained.

If you do not plan to use the instance again for a long time, right-click on the example in Figure 11.16 and select "Image" $\rightarrow$ "Create" to create an image of the instance. Then, select "Instance State" $\rightarrow$ "Terminate" to terminate the instance (it will no longer be billed for hard disk space). The next time you want to use this instance, you can follow the steps for creating and running an EC2 instance described in this section to create an instance based on the saved image. The only difference is that, in "1. Choose AMI" shown in Figure 11.10, you must use the "My AMIs" option on the left to select your saved image. The created instance will retain the information stored on the image hard disk. 

## Summary

* You can use cloud computing services to obtain more powerful computing resources and use them to run the deep learning code in this document.

## Problem

* The cloud offers convenience, but it does not come cheap. Research the prices of cloud services and find ways to reduce overhead.

## Discuss on our Forum

<div id="discuss" topic_id="2399"></div>
