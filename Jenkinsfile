stage("Build and Publish") {
  def  CONDA_ENV_NAME = "d2l-en-numpy"
  node {
    ws('workspace/${CONDA_ENV_NAME}') {
      sh "echo ${CONDA_ENV_NAME}"
    }
  }
}
