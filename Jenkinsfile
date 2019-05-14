stage("Build and Publish") {
  def TASK = "d2l-en-numpy"
  node {
    ws('workspace/${TASK}') {
      def ENV_NAME = "${TASK}-${EXECUTOR_NUMBER}"
      checkout scm

      sh '''set -ex
      echo "${TASK}"
      echo "${ENV_NAME}"
      '''
    }
  }
}
