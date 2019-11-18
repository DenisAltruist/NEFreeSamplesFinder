#!/usr/bin/env bash

apt-get install python
pip install cvxopt
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 60 --slave /usr/bin/g++ g++ /usr/bin/g++-8
