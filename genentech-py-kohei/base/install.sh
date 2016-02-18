#!/bin/bash

apt-get -y update
apt-get -y install wget vim zsh bzip2 build-essential git tree

chsh root -s /bin/zsh
source ~root/.zshrc

# Install anaconda2
wget https://3230d63b5fc54e62148e-c95ac804525aac4b6dba79b00b39d1d3.ssl.cf1.rackcdn.com/Anaconda2-2.4.0-Linux-x86_64.sh -O /root/anaconda2.sh
bash /root/anaconda2.sh -b
rm /root/anaconda2.sh

# Install python packages
conda upgrade --prefix /root/anaconda2 anaconda -y
pip install --upgrade pip
pip install awscli
pip install ml_metrics
pip install bloscpack
pip install blosc==1.2.5

# Build xgboost
git clone https://github.com/dmlc/xgboost
cd xgboost
git checkout 0.47
make

# Install python binding for xgboost
cd python-package/
python setup.py install

rm ~root/install.sh
