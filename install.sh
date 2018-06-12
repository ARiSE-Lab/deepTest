#!/bin/bash
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install build-essential cmake pkg-config
sudo apt-get install libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libxvidcore-dev libx264-dev
sudo apt-get install libgtk-3-dev
sudo apt-get install libatlas-base-dev gfortran
sudo apt-get install python2.7-dev python3.5-dev
sudo apt-get install unzip
wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
sudo python -m pip install -U pip == 8.0.1
pip install --user numpy
sudo apt-get install python-opencv
pip install --user scipy
pip install --user scikit-learn
pip install --user pillow
pip install --user h5py
pip install --user Theano==0.9
pip install --user tensorflow
pip install --user keras==1.2.2
pip install --user scikit-image
pip install --user objgraph
sudo apt-get install python-tk
