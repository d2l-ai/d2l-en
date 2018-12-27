stage("Sanity Check") {
  node {
    ws('workspace/d2l-en') {
      checkout scm
      sh "build/sanity_check.sh"
    }
  }
}

stage("Publish") {
  node {
    ws('workspace/d2l-en') {
      sh """#!/bin/bash
      set -ex
      if [[ ${env.BRANCH_NAME} == master ]]; then
         build/utils/upload_notebooks_no_output_github.sh . https://github.com/d2l-ai/notebooks
      fi
      """
    }
  }
}
