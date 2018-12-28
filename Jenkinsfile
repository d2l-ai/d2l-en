stage("Build and Publish") {
  node {
    ws('workspace/d2l-en') {
      checkout scm
      sh "build/sanity_check.sh"
      sh "build/clean_build.sh"
      sh "build/update_env.sh"
      if (env.BRANCH_NAME == 'master') {
        sh "build/publish_website.sh"
        withCredentials([usernamePassword(credentialsId: '13cb1009-3cda-49ef-9aaa-8a705fdaaeb7',
                         passwordVariable: 'GIT_PASSWORD', usernameVariable: 'GIT_USERNAME')]) {
          sh "build/publish_notebook.sh"
        }
      }
	}
  }
}
