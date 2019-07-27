#!/bin/sh

mkdir -p data/proc/new/224/0
mkdir -p data/proc/new/299/0
mkdir -p data/proc/old/224/0
mkdir -p data/proc/old/299/0
mkdir -p data/proc/aug/224/0
mkdir -p data/proc/aug/299/0
mkdir data/proc/new/224/1
mkdir data/proc/new/299/1
mkdir data/proc/old/224/1
mkdir data/proc/old/299/1
mkdir data/proc/aug/224/1
mkdir data/proc/aug/299/1
mkdir data/proc/new/224/2
mkdir data/proc/new/299/2
mkdir data/proc/old/224/2
mkdir data/proc/old/299/2
mkdir data/proc/aug/224/2
mkdir data/proc/aug/299/2
mkdir data/proc/new/224/3
mkdir data/proc/new/299/3
mkdir data/proc/old/224/3
mkdir data/proc/old/299/3
mkdir data/proc/aug/224/3
mkdir data/proc/aug/299/3
mkdir data/proc/new/224/4
mkdir data/proc/new/299/4
mkdir data/proc/old/224/4
mkdir data/proc/old/299/4
mkdir data/proc/aug/224/4
mkdir data/proc/aug/299/4

mkdir -p data/vectors/mobilenet
mkdir data/vectors/nasnet
mkdir data/vectors/inception-resnet

echo "Make sure the kaggle CLI is installed"

kaggle d download tanlikesmath/diabetic-retinopathy-resized
unzip diabetic-retinopathy-resized.zip
echo "rename resized_train_cropped to data/old, trainLabels_cropped.csv to data/old.csv and delete all else"

kaggle competitions download -c aptos2019-blindness-detection
unzip aptos2019-blindness-detection.zip
echo "rename train_images to data/new, and train.csv to data/new.csv"
