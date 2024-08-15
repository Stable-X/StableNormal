#!/bin/bash
set -euxo pipefail

export TZ=America/Los_Angeles
echo $TZ >/etc/timezone
ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

apt update
apt --yes full-upgrade
apt install --yes wget curl git zsh tmux htop vim ccache subversion gdb build-essential unzip \
	software-properties-common locales tzdata apt-transport-https openssh-server libgtest-dev \
	python3-pip libzstd-dev
# for opencv
 apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

wget https://apt.llvm.org/llvm.sh && \
    bash llvm.sh 15 && \
    ln -s /usr/bin/llvm-as-15 /usr/bin/llvm-as && \
    ln -s /usr/bin/llvm-config-15 /usr/bin/llvm-config && \
    rm llvm.sh

wget https://cmake.org/files/v3.26/cmake-3.26.0-linux-x86_64.tar.gz \
    && tar -zxvf cmake-3.26.0-linux-x86_64.tar.gz \
    && mv cmake-3.26.0-linux-x86_64/ /opt/cmake-3.26.0 \
    && ln -sf /opt/cmake-3.26.0/bin/* /usr/bin \
    && rm  cmake-3.26.0-linux-x86_64.tar.gz

apt install -y pkg-config libmysqlclient-dev

apt clean
rm -rf /var/lib/apt/lists/*

locale-gen "en_US.UTF-8"

git config --global --add safe.directory "*"
