@Library('blossom-github-lib@master')
import ipp.blossom.*

// Define the build containers for Slurm testing
def buildContainers = [
    'pytorch-25.02': [
        containerName: 'pytorch-2502',
        image: 'harbor.mellanox.com/ucx/x86_64/pytorch:25.02-py3',
        nodeSelector: 'kubernetes.io/os: linux'
    ],
    'pytorch-24.10': [
        containerName: 'pytorch-2410',
        image: 'harbor.mellanox.com/ucx/x86_64/pytorch:24.10-py3',
        nodeSelector: 'kubernetes.io/os: linux'
    ]
]

// Create container templates based on build containers
def containerTemplates = buildContainers.collect { name, config ->
    containerTemplate(name: config.containerName, image: 'golang:1.12.9', ttyEnabled: true, command: 'cat')
}

// Select default nodeSelector
def defaultNodeSelector = buildContainers.values()[0].nodeSelector

podTemplate(cloud:'il-ipp-blossom-prod', yaml : """
  apiVersion: v1
  kind: Pod
  spec:
    nodeSelector:
      ${defaultNodeSelector}""",
  containers: containerTemplates) {
      node(POD_LABEL) {
          def githubHelper

          stage('Get Token') {
              withCredentials([usernamePassword(credentialsId: 'github-token', passwordVariable: 'GIT_PASSWORD', usernameVariable: 'GIT_USERNAME')]) {
                  // create new instance of helper object
                  githubHelper = GithubHelper.getInstance("${GIT_PASSWORD}", githubData)
              }
          }

          def stageName = ''
          try {
              currentBuild.description = githubHelper.getBuildDescription()
              stageName = 'Code checkout'
              stage(stageName) {
                  if("Open".equalsIgnoreCase(githubHelper.getPRState())){
                    println "PR State is Open"
                    // checkout head of pull request
                    checkout changelog: true, poll: true, scm: [$class: 'GitSCM', branches: [[name: "pr/"+githubHelper.getPRNumber()]],                   doGenerateSubmoduleConfigurations: false,
                    submoduleCfg: [],
                    userRemoteConfigs: [[credentialsId: 'github-token', url: githubHelper.getCloneUrl(), refspec: '+refs/pull/*/head:refs/remotes/origin/pr/*']]]
                  }
                  else if("Merged".equalsIgnoreCase(githubHelper.getPRState())){
                    println "PR State is Merged"
                    // use following if you want to build merged code of the head & base branch
                    // ref : https://developer.github.com/v3/pulls/
                    checkout changelog: true, poll: true, scm: [$class: 'GitSCM', branches: [[name: githubHelper.getMergedSHA()]],
                    doGenerateSubmoduleConfigurations: false,
                    submoduleCfg: [],
                    userRemoteConfigs: [[credentialsId: 'github-token', url: githubHelper.getCloneUrl(), refspec: '+refs/pull/*/merge:refs/remotes/origin/pr/*']]]
                  }
              }

              def testResults = [:]

              stageName = 'Slurm Test'
              stage(stageName) {
                  // Run slurm tests in parallel for each container
                  parallel buildContainers.collectEntries { name, config ->
                      ["Slurm test with ${name}": {
                          container(config.containerName) {
                              println "Running Slurm test with ${name} container"
                              // Setup SSH credentials for svc-nixl
                              withCredentials([sshUserPrivateKey(credentialsId: 'svc-nixl', keyFileVariable: 'SSH_KEY_FILE', usernameVariable: 'SSH_USER', passphraseVariable: '')]) {
                                  sh(script: '''
                                  mkdir -p ~/.ssh
                                  cp $SSH_KEY_FILE ~/.ssh/id_rsa
                                  chmod 600 ~/.ssh/id_rsa
                                  ''', label: 'Setup SSH credentials for svc-nixl')
                                  sh """
                                     ./contrib/slurm/slurm_test.sh '.gitlab/test_python.sh /opt/nixl && .gitlab/test_cpp.sh /opt/nixl' -i ${config.image}
                                  """
                              }
                              testResults[name] = true
                          }
                      }]
                  }
              }

              // Report test results
              stage('Results') {
                  def failedTests = testResults.findAll { container, success -> !success }

                  if (failedTests) {
                      error "Failed tests: ${failedTests.keySet()}"
                  }
              }

              githubHelper.uploadLogs(this, env.JOB_NAME, env.BUILD_NUMBER, null, null)
          }
          catch (Exception ex){
              currentBuild.result = 'FAILURE'
              println ex
              githubHelper.uploadLogs(this, env.JOB_NAME, env.BUILD_NUMBER, null, null)
          }
      }
  }
