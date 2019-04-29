stage("Build and Publish") {
  node {
    ws('workspace/d2l-en') {
      checkout scm
      echo "Setup environment"
      sh '''set -ex
      env
      export CUDA_VISIBLE_DEVICES=$((EXECUTOR_NUMBER*2)),$((EXECUTOR_NUMBER*2+1))
      echo ${CUDA_VISIBLE_DEVICES}
      echo ${EXECUTOR_NUMBER}
      echo ${env.EXECUTOR_NUMBER}
      echo "Setup environment"
      '''
	}
  }
}
