stage("Build and Publish") {
  node {
    ws('workspace/d2l-en') {
      checkout scm
      echo "Setup environment"
      sh '''set -ex
      env
      echo ${EXECUTOR_NUMBER}
      echo ${env.EXECUTOR_NUMBER}
      echo "Setup environment"
      '''
	}
  }
}
