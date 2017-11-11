#!/bin/bash

SESSION_NAME="sc"

cd ~/Documents/code/github/sound_classification/src

tmux has-session -t ${SESSION_NAME}

if [ $? != 0 ]
then
    tmux new-session -s ${SESSION_NAME} -n vim -d
    
    # first vim
    tmux send-keys -t ${SESSION_NAME} 'vim' C-m
    tmux split-window -v -p 20
    #tmux split-window -h

    #tmux new-window -n 'jupyter' -t ${SESSION_NAME}
    #tmux send-keys -t ${SESSION_NAME}:1 'jupyter notebook' C-m

    #tmux select-window -t ${SESSION_NAME}:0
fi

tmux attach -t ${SESSION_NAME}
