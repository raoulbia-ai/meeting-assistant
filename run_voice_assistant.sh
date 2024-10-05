#!/bin/bash

# # Path to your conda.sh file. Adjust this path as necessary for your system.
# CONDA_PATH="$HOME/opt/miniconda3/condabin/"

# # Source the conda.sh file to initialize conda in the shell
# source "$CONDA_PATH"

# # Activate your conda environment
# conda activate interviewcomp

# # Export environment variables from .env file
# export $(grep -v '^#' ./.env | xargs)

# Run the Python script
python ./voice_assistant.py