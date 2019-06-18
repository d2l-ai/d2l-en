stage("Build and Publish") {
  def TASK = "d2l-en"
  node {
    ws("workspace/${TASK}") {
      checkout scm
      def ENV_NAME = "${TASK}-${EXECUTOR_NUMBER}"
      def CUDA_VISIBLE_DEVICES="$((EXECUTOR_NUMBER*2)),$((EXECUTOR_NUMBER*2+1))"

      sh label: "Build Environment", script: """set -ex
      echo ${CUDA_VISIBLE_DEVICES}
      rm -rf ~/miniconda3/envs/${ENV_NAME}
      conda create -n ${ENV_NAME} pip -y
      conda activate ${ENV_NAME}
      pip install mxnet-cu100
      pip install git+https://github.com/d2l-ai/d2l-book
      python setup.py develop
      pip list
      """

      sh label: "Check Execution Output", script: """set -ex
      conda activate ${ENV_NAME}
      d2lbook build outputcheck
      """

      sh label: "Execute Notebooks", script: """set -ex
      conda activate ${ENV_NAME}
      d2lbook build eval
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
      # don't pack downloaded data into the pkg
      mv _build/eval/data _build/data_tmp
      cp -r data _build/eval
      d2lbook build html pkg
      rm -rf _build/eval/data
      mv _build/data_tmp _build/eval/data
      """

      if (env.BRANCH_NAME == 'master') {
        sh label:"Publish", script:"""set -ex
        conda activate ${ENV_NAME}
        d2lbook deploy html pdf pkg
      """
      }
	}
  }
}
