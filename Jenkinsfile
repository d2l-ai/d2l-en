stage("Sanity Check") {
  node {
    ws('workspace/d2l-en') {
      checkout scm
      sh build/sanity_check.sh
    }
  }
}


stage("Build HTML") {
  node {
    ws('workspace/d2l-en') {
      checkout scm
    }
  }
}

stage("Build PDF") {
  node {
    ws('workspace/d2l-en') {
      checkout scm
    }
  }
}

stage("Publish") {
  node {
    ws('workspace/d2l-en') {
      checkout scm
    }
  }
}
