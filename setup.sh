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

# Get the absolute path of the current directory
PROJECT_ROOT="$(pwd)"

# Add PYTHONPATH export to venv activation script if not already present
ACTIVATE_FILE="venv/bin/activate"
PYTHONPATH_LINE="export PYTHONPATH=\"$PROJECT_ROOT:\$PYTHONPATH\""
if ! grep -Fxq "$PYTHONPATH_LINE" "$ACTIVATE_FILE"; then
    echo "$PYTHONPATH_LINE" >> "$ACTIVATE_FILE"
    echo "Added PYTHONPATH to $ACTIVATE_FILE"
fi

# Activate the virtual environment
source venv/bin/activate

# Ensure pip is installed and install dependencies
pip install --upgrade pip
pip install -r requirements.txt