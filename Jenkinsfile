stage("Build and Publish") {
  node {
    ws('workspace/d2l-en') {
      checkout scm
      sh "git submodule update --init"
      sh "build/utils/sanity_check.sh"
      sh "build/utils/clean_build.sh"
      sh "conda env update -f build/env.yml"
      sh "build/utils/build_html.sh en"
      sh "build/utils/build_pdf.sh en"
      sh "build/utils/build_pkg.sh en"
      if (env.BRANCH_NAME == 'master') {
        sh "build/utils/publish_website.sh en"
        withCredentials([usernamePassword(credentialsId: 'ea9ac23b-cc0d-4d2c-a080-67d829e1415a',
                         passwordVariable: 'GIT_PASSWORD', usernameVariable: 'GIT_USERNAME')]) {
          sh "build/publish_notebook.sh"
        }
      }
	}
  }
}
