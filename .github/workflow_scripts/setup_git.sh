#!/bin/bash

function setup_git {
    # Retrieve the SSH key from environment variable
    GIT_SSH_KEY=$D2L_BOT_GITHUB

    # Write the SSH key to a file
    mkdir -p $HOME/.ssh
    echo "$GIT_SSH_KEY" > $HOME/.ssh/id_rsa
    chmod 600 $HOME/.ssh/id_rsa

    # Set up SSH config to use the key
    echo "Host github.com\n  IdentityFile $HOME/.ssh/id_rsa\n  StrictHostKeyChecking no\n" >> $HOME/.ssh/config

    git config user.name "d2l-bot"
}
