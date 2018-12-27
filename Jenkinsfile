stage("Sanity Check") {
  node {
    ws('workspace/d2l-en') {
      checkout scm
      sh "build/sanity_check.sh"
    }
  }
}

stage("Build and Publish") {
  node {
    ws('workspace/d2l-en') {
	  checkout scm
      sh """#!/bin/bash
      set -e
      git submodule update --init
      conda activate d2l-en-build
      if [[ ${env.BRANCH_NAME} == master ]]; then
         build/utils/upload_notebooks_no_output_github.sh . https://github.com/d2l-ai/notebooks
      fi
      """
	}
  }
}
