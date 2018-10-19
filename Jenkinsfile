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
          aws s3 sync --delete build/_build/html/ s3://diveintodeeplearning.org/ --acl public-read
      else
          aws s3 sync --delete build/_build/html/ s3://diveintodeeplearning-staging/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/ --acl public-read
      fi
      """

      if (env.BRANCH_NAME.startsWith("PR-")) {
        pullRequest.comment("Job ${env.BRANCH_NAME}-${env.BUILD_NUMBER} is done. \nDocs are uploaded to http://diveintodeeplearning-staging.s3-website-us-west-2.amazonaws.com/${env.BRANCH_NAME}/${env.BUILD_NUMBER}/index.html")
      }
    }
  }
}
