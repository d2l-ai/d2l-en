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
      checkout scm
      sh """#!/bin/bash
      set -ex
      aws s3 sync --delete build/_build/html/ s3://diveintodeeplearning.org/ --acl public-read
      """
    }
  }
}
