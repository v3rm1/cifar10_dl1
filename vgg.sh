#!/bin/bash
mkdir data
mkdir models
mkdir logs
mkdir logs/vgg
mkdir models/baseline
mkdir models/batchnorm
mkdir models/dropout
mkdir models/w_decay
mkdir logs/vgg/baseline
mkdir logs/vgg/batchnorm
mkdir logs/vgg/dropout
mkdir logs/vgg/w_decay

pip3 --no-cache-dir install -r requirements.txt

python3 vgg_app.py