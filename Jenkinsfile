stage("Build and Publish") {
  node {
    ws('workspace/d2l-en') {
      checkout scm
      sh "build/sanity_check.sh"
      sh "build/clean_build.sh"
      if (env.BRANCH_NAME == 'master') {
        sh "build/utils/upload_doc_s3.sh build/_build/html/ s3://en.d2l.ai"
        withCredentials([usernamePassword(credentialsId: '13cb1009-3cda-49ef-9aaa-8a705fdaaeb7',
                         passwordVariable: 'GIT_PASSWORD', usernameVariable: 'GIT_USERNAME')]) {
          sh "build/publish_notebook.sh"
        }
      }
	}
  }
}
