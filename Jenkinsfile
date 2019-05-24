stage("Build and Publish") {
  node {
    ws('workspace/d2l-en') {
      checkout scm

      sh label: "Build Environment", script: '''set -ex
      rm -rf ~/miniconda3/envs/d2l-en-build-${EXECUTOR_NUMBER}
      conda create -n d2l-en-build-${EXECUTOR_NUMBER} pip -y
      conda activate d2l-en-build-${EXECUTOR_NUMBER}
      pip install mxnet-cu100
      pip install git+https://github.com/d2l-ai/d2l-book
      # pip install d2l>=0.9.2
      python setup.py develop
      pip list
      '''

      sh label: "Check Execution Output", script: '''set -ex
      conda activate d2l-en-build-${EXECUTOR_NUMBER}
      d2lbook build outputcheck
      '''

      sh label: "Execute Notebooks", script: '''set -ex
      conda activate d2l-en-build-${EXECUTOR_NUMBER}
      export CUDA_VISIBLE_DEVICES=$((EXECUTOR_NUMBER*2)),$((EXECUTOR_NUMBER*2+1))
      d2lbook build eval
      '''

      // sh '''set -ex
      // conda activate d2l-en-build-${EXECUTOR_NUMBER}
      // d2lbook build linkcheck
      // '''

      sh label:"Build HTML", script:'''set -ex
      conda activate d2l-en-build-${EXECUTOR_NUMBER}
      ./static/build_html.sh
      '''

      sh label:"Build PDF", script:'''set -ex
      conda activate d2l-en-build-${EXECUTOR_NUMBER}
      d2lbook build pdf
      '''

      sh label:"Build Package", script:'''set -ex
      conda activate d2l-en-build-${EXECUTOR_NUMBER}
      # don't pack downloaded data into the pkg
      mv _build/eval/data _build/data_tmp
      cp -r data _build/eval
      d2lbook build html pkg
      mv _build/data_tmp _build/eval/data
      '''

      if (env.BRANCH_NAME == 'master') {
        sh label:"Publish", script:'''set -ex
        conda activate d2l-en-build-${EXECUTOR_NUMBER}
        d2lbook deploy html pdf pkg
      '''
      }
	}
  }
}
