#!/bin/bash

echo "pip recently upgrade..."
pip install --upgrade pip

echo "cuda 11.3 pytorch install..."
pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html


