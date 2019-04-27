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
	}
  }
}
