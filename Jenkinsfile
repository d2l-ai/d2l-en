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
      echo "Build HTML"
      sh '''set -ex
      conda activate d2l-en-build
      rm -rf _build/rst
      d2lbook build rst
      # copy frontpage
      cp frontpage/frontpage.html _build/rst/
      d2lbook build html
      cp frontpage/_images/* _build/html/_images/
      '''
      echo "Build PDF"
      sh '''set -ex
      conda activate d2l-en-build
      d2lbook build pdf
      '''
      echo "Build Package"
      sh '''set -ex
      conda activate d2l-en-build
      # don't pack downloaded data into the pkg
      rm -rf _build/eval/*.zip
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
