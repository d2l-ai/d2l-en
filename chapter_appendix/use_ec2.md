# Run on an EC2 instance 
This chapter shows, how to allocate a CPU/GPU instance in AWS and how to setup the Deep Learning environment.

We first need [an AWS account](https://aws.amazon.com/), and then go the EC2 console
after login in.

<img src="img/aws.png" width="400"/>

Then click "launch instance" to select the operation system and instance type.

<img src="img/ec2.png" width="500"/>

AWS offers [Deep Learning AMIs](https://docs.aws.amazon.com/dlami/latest/devguide/options.html) 
that come with the latest versions of Deep Learning frameworks. The Deep Learning AMIs provide 
all necessary packages and drivers and allow you to directly start implementing 
and training your models. Deep Learning AMIs use use binaries that are optimized to run on AWS instances to accelerate model training and inference.
In this tutorial we use Deep Learning AMI (Ubuntu) Version 19.0:


<img src="img/os.png" width="600"/>


We choose "p2.xlarge", which contains a single Nvidia K80 GPU. Note that there is a
large number of instance, refer to
[ec2instances.info](http://www.ec2instances.info/) for detailed configurations
and fees.

<img src="img/p2x.png" width="400"/>

Note that we need to check the instance limits to guarantee that we can request
the resource. If running out of limits, we can request more capacity by clicking
the right link, which often takes about a single workday to process.

<img src="img/limits.png" width="500"/>

On the next step we increased the disk from 8 GB to 40 GB so we have enough
space store a reasonable size dataset. For large-scale
datasets, we can "add new volume". Also you selected a very powerful GPU
instance such as "p3.8xlarge", make sure you selected "Provisioned IOPS" in the
volume type for better I/O performance.

<img src="img/disk.png" width="500"/>

Then we launched with other options as the default values. The last step before
launching is choosing the ssh key, you may need to generate and store a key if
you don't have one before.

<img src="img/keypair.png" width="400"/>

After clicked "launch instances", we can check the status by clicking the
instance ID link.

<img src="img/launching.png" width="500"/>

Once the status is green, we can right-click and select "connect" to get the access instruction.

<img src="img/connect.png" width="500"/>

With the given address, we can log into our instance:

<img src="img/login_screen.png" width="700"/>

The login screen will show a long list of available conda environments for the different Deep Learning frameworks, CUDA driver and Python versions. With ```conda activate``` you can easily switch into the different environments. In the following example we switch to the MXNet Python 3.6 environment:

<img src="img/mxnet.png" width="400"/> 

Now you are ready to start developing and training MXNet models. Once you start training, you can check the GPU status with ```nividia-smi```.
