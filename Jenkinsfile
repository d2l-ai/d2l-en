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
      tik=\`date +%s\`

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

	  tok=\`date +%s\`
	  runtime=$((end-start))
      convertsecs() {
		((h=${1}/3600))
		((m=(${1}%3600)/60))
		((s=${1}%60))
		printf "%02d:%02d:%02d\n" $h $m $s
	  }
	  echo $(convertsecs $runtime)
	}
  }
}

