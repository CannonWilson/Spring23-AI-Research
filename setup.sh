#!/bin/bash

# This file completes the setup processes 
# necessary to run the files in 
# this project after it is cloned. 
# This file assumes python is downloaded.

# Make sure pip is installed & up to date, then install venv
python -m ensurepip --upgrade
python -m pip install --user --upgrade pip
python -m pip install --user virtualenv

# Start by creating/activating the env
python -m venv research-env
source research-env/bin/activate

# Pip install packages into env
pip install -r requirements.txt

# Install CelebA dataset locally
gdown "https://drive.google.com/uc?id=1HpoLLP9x7ON5nn5TnC7CPf2uoRnO2VnD"

# Create and fill the train/val/test directories
python make_dataset.py

echo "### Thanks for downloading, setup is now complete. ###"