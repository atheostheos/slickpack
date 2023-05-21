#!/usr/bin/env bash

docker run -it \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix/:/tmp/.X11-unix/ \
  -v $HOME/.Xauthority:/root/.Xauthority \
  -e DISPLAY=$DISPLAY \
  --gpus all \
  --device /dev/video0 \
  -v /home/atheostheos/git/slickpack:/slickpack \
  -t slickpack:latest
