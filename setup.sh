#!/bin/bash

# Ensure python3-venv is installed
sudo apt update
sudo apt install -y python3.10-venv

# Remove existing virtual environment if any
rm -rf venv

# Create a new virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Ensure pip is installed and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
