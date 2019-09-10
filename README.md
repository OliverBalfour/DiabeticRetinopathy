
# Diabetic Retinopathy Diagnosis

## Install & Run
Install `conda`, then:

```sh
conda install python=3.7
conda create --name PythonGPU
conda activate PythonGPU
conda install -c anaconda numpy jupyter matplotlib seaborn pandas scikit-learn tensorflow-gpu cuda-toolkit=9.0 py-xgboost-gpu
pip3 install opencv-python #OR conda install -c menpo opencv
```

For Jupyter notebook (not necessary):
```sh
conda install notebook ipykernel
ipython kernel install --user --name=PythonGPU
jupyter notebook
```
Then in the notebook editor click kernel->change kernel->PythonGPU.

## Data sources

APTOS Kaggle competition:

 - train_images (data/new), test_images, train.csv (data/new.csv), test.csv

Google Diabetic Retinopathy competition (resized by tanlovesmath)

 - resized_train_cropped (data/old), trainLabels_cropped.csv (data/old.csv)

All data is preprocessed and stored in data/proc/{size \\in {224,299}}/{class \\in {0,1,2,3,4}}

## Running

Download data, preprocess, augment, train models, train stacked models, etc. This section may be slightly out of date.

0. `setup.sh` to download data and generate filesystem
0. `python3 processing/preprocess.py` to preprocess it
0. `python3 processing/corruption-checker.py` and delete all corrupt images
0. `python3 processing/augment.py` to augment it and generate datasets
0. `python3 models/densenet.py` to train densenet121
0. `python3 models/mobilenet.py` to train mobilenet
0. `python3 models/extraction/*.py` to extract features using all CNNs
0. `python3 models/ensemble/model-generation.py` to train other models stacked on CNN filters
0. `python3 models/ensemble/model-selection.py` to choose best models
0. `python3 models/ensemble/model-evaluation.py` to compute predictions for train and test data
0. `python3 models/ensemble/average.py` to compute ensemble outputs
0. `python3 models/ensemble/results-eda.py` to generate confusion matrices

TODO: Once the pipeline is complete, run `python3 models/evaluate.py filename.png/jpg` to get a prediction for a single image.
