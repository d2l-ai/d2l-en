stage("Build and Publish") {
  def TASK = "d2l-en-numpy"
  node {
    ws('workspace/${TASK}') {
      def ENV_NAME = "${TASK}-${EXECUTOR_NUMBER}"
      checkout scm

      sh "echo ${TASK}"
      sh "echo ${ENV_NAME}"
    }
  }
}
