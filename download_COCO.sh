#!/bin/bash

echo "make data folder"
mkdir ./data
cd data

echo "Download tain2014..."
wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip
rm train2014.zip 

echo "Download val2014..."
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip
rm val2014.zip

echo "Download annotations..."
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip
rm annotations_trainval2014.zip
