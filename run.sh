#!/bin/bash

# Ensure the script exits on any error
set -e

# Array of filenames to pass as arguments
FILES=("testes_profile_exp12_simple_uniform.txt" "testes_profile_exp12_simple_norm.txt" "testes_profile_exp12_simple_geo.txt" "testes_profile_exp12_simple_porto.txt")

# Loop through each file and call the Python program
for FILE in "${FILES[@]}"; do
    echo "Running ALOG with argument: $FILE"
    python3 alog.py "$FILE"
done

echo "All executions completed."
