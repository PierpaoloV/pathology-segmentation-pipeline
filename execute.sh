#!/bin/bash

# Check if extra arguments were given and execute as a command.
if [ -z "$1" ]
    then
        /usr/sbin/sshd
        cd /home/user && sudo --set-home --preserve-env --user=user \
            /bin/bash -c '/usr/local/bin/jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token='
    else
        echo "Execute command: ${@}"
        cd /home/user && sudo --user=user --set-home "${@}"
fi
