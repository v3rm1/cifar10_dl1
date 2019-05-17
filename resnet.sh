#!/bin/bash
mkdir data
mkdir models
mkdir logs
mkdir logs/res
mkdir models/baseline
mkdir models/batchnorm
mkdir models/dropout
mkdir models/w_decay
mkdir logs/res/baseline
mkdir logs/res/batchnorm
mkdir logs/res/dropout
mkdir logs/res/w_decay

pip3 --no-cache-dir install -r requirements.txt

python3 resnet_app.py