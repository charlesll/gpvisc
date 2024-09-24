#!/bin/bash

# Define the base name for the output directories
BASE_DIR="./models/GP/GP_model"

# Loop 100 times
for i in {0..100}
do
    # Define the output directory for the current iteration
    GP_SAVE_NAME="${BASE_DIR}${i}/"
    
    # Create the output directory if it doesn't exist
    mkdir -p $GP_SAVE_NAME
    
    # Run the Python script with the current output directory
    python 5_gp_train.py --gp_save_name $GP_SAVE_NAME --training_iter 60000 --early_criterion 5000 --training_iter_gp 1500
    
    # Optionally, you can add a sleep command to wait between iterations
    # sleep 1
done
