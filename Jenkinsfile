stage("Build and Publish") {
  def TASK = "d2l-en"
  def BRANCH = env.BRANCH_NAME
  node {
    ws("workspace/${TASK}") {
      checkout scm
      def ENV_NAME = "${TASK}-${EXECUTOR_NUMBER}";
      def EID = EXECUTOR_NUMBER.toInteger()
      def CUDA_VISIBLE_DEVICES=(EID*2).toString() + ',' + (EID*2+1).toString();

      sh label: "Build Environment", script: """set -ex
      echo ${ENV_NAME}
      echo ${BRANCH}

      """
    }
  }
}
