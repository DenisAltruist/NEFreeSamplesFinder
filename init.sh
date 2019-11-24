#!/usr/bin/env bash


apt-get update \
    && apt-get install -y software-properties-common \
    && apt-add-repository -y ppa:ubuntu-toolchain-r/test

apt-get update && apt-get install -y \
gcc-8 \
  g++-8 \
  python-dev \
  python-setuptools \
  python-pip \
  python3-pip

apt-get install python
pip install cvxopt
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 60 --slave /usr/bin/g++ g++ /usr/bin/g++-8

python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose