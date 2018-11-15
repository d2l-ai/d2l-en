stage("Sanity Check") {
  node {
    ws('workspace/d2l-en') {
      checkout scm
      sh "build/sanity_check.sh"
    }
  }
}


stage("Build HTML") {
  node {
    ws('workspace/d2l-en') {
      checkout scm
      sh "build/build_html.sh"
    }
  }
}

stage("Build PDF") {
  node {
    ws('workspace/d2l-en') {
      checkout scm
      sh "build/build_pdf.sh"
    }
  }
}

stage("Publish") {
  node {
    ws('workspace/d2l-en') {
      sh """#!/bin/bash
      set -ex
      if [[ ${env.BRANCH_NAME} == master ]]; then
          conda activate d2l-en-build
          aws s3 sync --delete build/_build/html/ s3://diveintodeeplearning.org/ --acl public-read
      fi
      """
    }
  }
}
