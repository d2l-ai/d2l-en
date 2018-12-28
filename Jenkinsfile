stage("Build and Publish") {
  node {
    ws('workspace/d2l-en') {
      checkout scm
      sh "build/sanity_check.sh"
      sh "build/clean_build.sh"
      sh "build/update_env.sh"
      sh "build/build_html.sh"
      sh "build/build_pdf.sh"
      sh "build/build_pkg.sh"
      when {
        branch "master"
      }
      steps {
        sh "build/utils/upload_doc_s3.sh build/_build/html/ s3://en.d2l.ai"
        withCredentials([usernamePassword(credentialsId: '13cb1009-3cda-49ef-9aaa-8a705fdaaeb7',
                         passwordVariable: 'GIT_PASSWORD', usernameVariable: 'GIT_USERNAME')]) {
          sh "build/publish_notebook.sh"
        }
      }
	}
  }
}
