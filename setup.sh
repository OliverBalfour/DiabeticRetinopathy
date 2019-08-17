#!/bin/sh

echo "Making data directories"

mkdir -p data/proc/new/224/0
mkdir -p data/proc/new/224/1
mkdir -p data/proc/new/224/2
mkdir -p data/proc/new/224/3
mkdir -p data/proc/new/224/4

mkdir -p data/proc/old/224/0
mkdir -p data/proc/old/224/1
mkdir -p data/proc/old/224/2
mkdir -p data/proc/old/224/3
mkdir -p data/proc/old/224/4

mkdir -p data/proc/severity/train/224/0
mkdir -p data/proc/severity/train/224/1
mkdir -p data/proc/severity/train/224/2
mkdir -p data/proc/severity/train/224/3
mkdir -p data/proc/severity/train/224/4

mkdir -p data/proc/binary/train/224/0
mkdir -p data/proc/binary/train/224/1

mkdir -p data/proc/binary/test/224/0
mkdir -p data/proc/binary/test/224/1

mkdir -p data/vectors/mobilenet
mkdir -p data/vectors/mobilenet-test
mkdir -p data/vectors/mobilenet-cat
mkdir -p data/vectors/mobilenet-cat-test
mkdir -p data/vectors/densenet
mkdir -p data/vectors/densenet-test

echo "Done making directories"
echo "Make sure the kaggle CLI is installed, downloading and unzipping now:"

kaggle d download tanlikesmath/diabetic-retinopathy-resized
unzip diabetic-retinopathy-resized.zip
echo "rename resized_train_cropped to data/old, trainLabels_cropped.csv to data/old.csv and delete all else"

kaggle competitions download -c aptos2019-blindness-detection
unzip aptos2019-blindness-detection.zip
echo "rename train_images to data/new, and train.csv to data/new.csv"
