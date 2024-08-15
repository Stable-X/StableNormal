IMG=stable-normal-dev:py311_cu121
REPO_NAME=StableNormal
NAME=${USER}_${REPO_NAME}


docker run -it -d --name $NAME \
  --gpus "device=1" \
  --hostname in_docker \
  --add-host in_docker:127.0.0.1 \
  --add-host $(hostname):127.0.0.1 \
  --shm-size 2G \
  -e DISPLAY \
  -p 6002:22 \
  -v /etc/localtime:/etc/localtime:ro \
  -v /media:/media \
  -v /mnt:/mnt \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ${PWD}:/${REPO_NAME} \
  -w /${REPO_NAME} \
  $IMG \
  /bin/bash
