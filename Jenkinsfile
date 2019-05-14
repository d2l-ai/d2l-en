stage("Build and Publish") {
  def TASK = "d2l-en-numpy"
  def ENV_NAME = "${TASK}-${EXECUTOR_NUMBER}"
  node {
    ws('workspace/${TASK}') {
      checkout scm

      sh label: "Build Environment", script: '''set -ex
      rm -rf ~/miniconda3/envs/${ENV_NAME}
      conda create -n ${ENV_NAME} pip -y
      conda activate ${ENV_NAME}
      pip install https://s3-us-west-2.amazonaws.com/apache-mxnet/dist/python/numpy/20190514/mxnet-1.5.0b20190514-py2.py3-none-manylinux1_x86_64.whl
      pip install d2l>=0.9.2
      pip install git+https://github.com/d2l-ai/d2l-book
      pip list
      """
    }
  }
}
