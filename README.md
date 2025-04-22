# Overview

These scripts allow for training random forest models on CPTV thermal images. The model created can be run on a single image to detect if it is false-positive or animal

## Extracting features

First download the CPTV files by running (https://github.com/TheCacophonyProject/cptv-download)

`python cptv-download.py <dir> <user> <password>`

Extract the features

`python3 forestmodel.py <dir>`

This will create a numpy file features.npy which can be used to train

## Training a model

`python3 validate.py`

This will train a model against the extracted features in `features.npy` and save it as `model.pkl` and associated metadata in `model.json`

## Evaluate

The evaluate script can be used to see a confusion matrix against a folder of CPTV files or `features.npy` file

`python3 --cptv_dir "../test-cptv-files" evalute.py model.pkl confusion.png`
