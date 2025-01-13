#!/bin/bash

# Desired Python version
PYTHON_VERSION="3.10.2"

# Check if Python 3.10.2 is installed
if ! python3.10 --version &>/dev/null; then
    echo "Python 3.10.2 is not installed. Please install it first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    python3.10 -m venv .venv
    echo "Virtual environment created with Python $PYTHON_VERSION."
fi

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
if [ -f "common/requirements.txt" ]; then
    python3.10 -m pip install --upgrade pip
    python3.10 -m pip install -r common/requirements.txt
    echo "Dependencies installed from common/requirements.txt."
    pre-commit install
else
    echo "No common/requirements.txt file found. Skipping dependency installation."
fi

echo "Setup complete. Virtual environment is activated."

