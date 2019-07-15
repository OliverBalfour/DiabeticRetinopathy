
# Diabetic Retinopathy Diagnosis

## Install & Run
Install `conda`, then:

`conda install python=3.7`
`conda create --name PythonGPU`
`conda activate PythonGPU`
`conda install -c anaconda numpy jupyter matplotlib seaborn pandas scikit-learn tensorflow-gpu cuda-toolkit=9.0`
`conda install notebook ipykernel`
`ipython kernel install --user --name=PythonGPU`
`jupyter notebook`
`pip3 install opencv-python OR conda install -c menpo opencv`

Then in the notebook editor click kernel->change kernel->PythonGPU.

## Data sources

APTOS Kaggle competition:
 - train_images (data/new), test_images, train.csv (data/new.csv), test.csv

Google Diabetic Retinopathy competition (resized by tanlovesmath)
 - resized_train_cropped (data/old), trainLabels_cropped.csv (data/old.csv)

All data is preprocessed and stored in data/proc/{class \\in {0,1,2,3,4}}

To download and process data, run `setup.sh` and then `python3 processing/bulk-preprocess.py`
