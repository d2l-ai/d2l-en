stage("Build and Publish") {
  node {
    ws('workspace/d2l-en') {
      checkout scm
      echo "Setup environment"
      sh '''set -ex
      conda remove -n d2l-en-build --all -y
      conda create -n d2l-en-build pip -y
      conda activate d2l-en-build
      pip install mxnet-cu100
      pip install d2lbook>=0.1.5 d2l>=0.9.2
      '''
      echo "Execute notebooks"
      sh '''set -ex
      conda activate d2l-en-build
      d2lbook build eval
      '''
      echo "Build HTML/PDF/Pakage"
      sh '''set -ex
      conda activate d2l-en-build
      d2lbook build html pdf
      mv _build/eval/data _build/data_tmp
      cp -r data _build/eval
      d2lbook build html pkg
      mv _build/data_tmp _build/eval/data
      '''
      if (env.BRANCH_NAME == 'master') {
        echo "Publish"
        sh '''set -ex
        conda activate d2l-en-build
        d2lbook deploy html pdf pkg
      '''
      }
	}
  }
}
