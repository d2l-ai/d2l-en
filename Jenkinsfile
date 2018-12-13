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
      sh "build/build_html.sh"
      sh "build/build_pdf.sh"
      sh "build/build_pkg.sh"
      sh """#!/bin/bash
      set -ex
      if [[ ${env.BRANCH_NAME} == master ]]; then
          build/upload.sh
      fi
      """
	}
  }
}

