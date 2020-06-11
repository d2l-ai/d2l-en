stage("Build and Publish") {
  def TASK = "d2l-face"
  node {
    ws("workspace/${TASK}") {
      checkout scm
      def ENV_NAME = "${TASK}-${EXECUTOR_NUMBER}";
      def EID = EXECUTOR_NUMBER.toInteger()
      def CUDA_VISIBLE_DEVICES=(EID*2).toString() + ',' + (EID*2+1).toString();

      sh label: "Build Environment", script: """set -ex
      rm -rf ~/miniconda3/envs/${ENV_NAME}
      conda create -n ${ENV_NAME} pip python=3.7 -y
      conda activate ${ENV_NAME}
      # d2l
      python setup.py develop
      # mxnet
      pip install mxnet-cu101==2.0.0b20200528 -f https://dist.mxnet.io/python/all
      pip install git+https://github.com/d2l-ai/d2l-book
      # pytorch
      # pip install torch==1.5.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
      # pip install torchvision
      # check
      pip list
      nvidia-smi
      """

      sh label: "Check Execution Output", script: """set -ex
      conda activate ${ENV_NAME}
      d2lbook build outputcheck
      """

      sh label: "Execute Notebooks", script: """set -ex
      conda activate ${ENV_NAME}
      export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
      export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
      ./static/cache.sh restore _build/eval/data _build/data_tmp
      ./static/clean_eval.sh
      d2lbook build eval
      ./static/cache.sh store _build/eval/data _build/data_tmp
      """

      sh label: "Execute Notebooks [Pytorch]", script: """set -ex
      conda activate ${ENV_NAME}
      export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
      export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
      ./static/cache.sh restore _build/eval_pytorch/data _build/data_pytorch_tmp
      d2lbook build eval --tab pytorch
      ./static/cache.sh store _build/eval_pytorch/data _build/data_pytorch_tmp
      """

      sh label:"Build HTML", script:"""set -ex
      conda activate ${ENV_NAME}
      ./static/build_html.sh
      """

      sh label:"Build PDF", script:"""set -ex
      conda activate ${ENV_NAME}
      d2lbook build pdf
      """

      sh label:"Build Package", script:"""set -ex
      conda activate ${ENV_NAME}
      d2lbook build pkg
      ./static/build_whl.sh
      """

      if (env.BRANCH_NAME == 'face') {
        sh label:"Publish", script:"""set -ex
        conda activate ${ENV_NAME}
        d2lbook deploy html pdf pkg
      """
      }
    }
  }
}
