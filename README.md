
# Diabetic Retinopathy Diagnosis

NOTE: This README is waaaay out of date. Code is provided as is and entirely undocumented. Below shell commands are provided for my reference only.

## Install & Run
Install `conda`, then:

```sh
conda install python=3.7
conda create --name PythonGPU
conda activate PythonGPU
conda install -c anaconda numpy jupyter matplotlib seaborn pandas scikit-learn tensorflow-gpu cuda-toolkit=9.0
conda install notebook ipykernel
ipython kernel install --user --name=PythonGPU
jupyter notebook
pip3 install opencv-python #OR conda install -c menpo opencv
```

Then in the notebook editor click kernel->change kernel->PythonGPU.

## Data sources

APTOS Kaggle competition:

 - train_images (data/new), test_images, train.csv (data/new.csv), test.csv

Google Diabetic Retinopathy competition (resized by tanlovesmath)

 - resized_train_cropped (data/old), trainLabels_cropped.csv (data/old.csv)

All data is preprocessed and stored in data/proc/{size \\in {224,299}}/{class \\in {0,1,2,3,4}}

## Running

Download data, preprocess, augment, train models, train stacked models, etc.

0. `setup.sh` to download data
0. `python3 processing/preprocess.py` to preprocess it
0. `python3 processing/augment.py` to augment it
0. `python3 processing/corruption-checker.py` and delete all corrupt images (in 224 and 299 dirs)
0. `python3 models/train/simple-cnn.py` to train main CNN
0. `python3 models/extraction/*.py` to extract features using all CNNs
0. `python3 models/train/*.py` to train other models stacked on CNN filters

TODO: Once the pipeline is complete, run `python3 models/evaluate.py filename.png/jpg` to get a prediction for a single image.
