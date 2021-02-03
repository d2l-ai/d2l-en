stage("Build and Publish") {
  def TASK = "d2l-jax"
  node {
    ws("workspace/${TASK}") {
      checkout scm
      def ENV_NAME = "${TASK}-${EXECUTOR_NUMBER}";

      sh label: "Build Environment", script: """set -ex
      rm -rf ~/miniconda3/envs/${ENV_NAME}
      conda create -n ${ENV_NAME} pip python=3.8 -y
      conda activate ${ENV_NAME}
      # d2l
      python setup.py develop
      # mxnet
      pip install mxnet-cu101==1.7.0
      pip uninstall -y d2lbook
      pip install git+https://github.com/d2l-ai/d2l-book
      # pytorch
      pip install --pre torch==1.8.0.dev20201219+cu101 -f https://download.pytorch.org/whl/nightly/cu101/torch_nightly.html
      pip install --pre torchvision==0.9.0.dev20201219+cu101 -f https://download.pytorch.org/whl/nightly/cu101/torch_nightly.html
      # tensorflow
      pip install tensorflow==2.3.1
      pip install tensorflow-probability==0.11.1
      # jax
      pip install https://storage.googleapis.com/jax-releases/cuda101/jaxlib-0.1.49-cp38-none-linux_x86_64.whl
      pip install jax
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
      export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
      ./static/cache.sh restore _build/eval/data
      d2lbook build eval
      ./static/cache.sh store _build/eval/data
      """

      sh label: "Execute Notebooks [Pytorch]", script: """set -ex
      conda activate ${ENV_NAME}
      export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
      ./static/cache.sh restore _build/eval_pytorch/data
      d2lbook build eval --tab pytorch
      ./static/cache.sh store _build/eval_pytorch/data
      """

      sh label: "Execute Notebooks [Tensorflow]", script: """set -ex
      conda activate ${ENV_NAME}
      export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
      ./static/cache.sh restore _build/eval_tensorflow/data
      d2lbook build eval --tab tensorflow
      ./static/cache.sh store _build/eval_tensorflow/data
      """

      sh label: "Execute Notebooks [Jax]", script: """set -ex
      conda activate ${ENV_NAME}
      export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
      ./static/cache.sh restore _build/eval_jax/data
      d2lbook build eval --tab jax
      ./static/cache.sh store _build/eval_jax/data
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

      if (env.BRANCH_NAME == 'jax') {
        sh label:"Publish", script:"""set -ex
        conda activate ${ENV_NAME}
        d2lbook deploy html pdf pkg
      """
      }
    }
  }
}
