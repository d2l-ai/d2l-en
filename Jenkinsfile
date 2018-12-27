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
      git submodule update --init
      withCredentials([usernamePassword(credentialsId: '13cb1009-3cda-49ef-9aaa-8a705fdaaeb7',
                       passwordVariable: 'GIT_PASSWORD', usernameVariable: 'GIT_USERNAME')]) {
        sh "build/publish_notebook.sh"
      }
	}
  }
}
