stage("Build and Publish") {
  node {
    ws('workspace/d2l-en') {
      checkout scm
      echo "Setup environment"
      sh '''set -ex
      conda remove -n d2l-en-build --all
      conda create -n d2l-en-build pip -y
      conda activate d2l-en-build
      pip install d2lbook>=0.1.5 d2l>=0.9.1 mxnet-cu100>=1.5.0b20190428
      '''
      echo "Execute notebooks"
      echo "Build HTML/PDF/Pakage"
      echo "Publish"
	}
  }
}
