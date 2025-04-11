#!/bin/bash

#SBATCH --job-name=transformer_en_de_ita_spa    # Job name
#SBATCH --output=/storage/work/mqf5675/Masters/NLP/project/model_output/transformer_en_de_%j.out        # Output file (%j is job ID)
#SBATCH --error=/storage/work/mqf5675/Masters/NLP/project/model_output/transformer_en_de_%j.err         # Error file (%j is job ID)
#SBATCH --partition=gpu                  # Partition name (change to your cluster's GPU partition)
#SBATCH --nodes=4                        # Number of nodes
#SBATCH --ntasks=4                       # Number of tasks
#SBATCH --cpus-per-task=8                # Number of CPU cores per task
#SBATCH --mem=32G                        # Memory per node (16GB)
#SBATCH --time=40:00:00                  # Time limit (HH:MM:SS)

#SBATCH --mail-type=ALL                  # Email notifications (ALL = BEGIN, END, FAIL)
#SBATCH --mail-user=mqf5675@psu.edu # Replace with your email

# Load necessary modules (if required by Roar)
source activate deep_learning           # Activate TensorFlow 2.8 environment (matches your pre-defined env)

# Navigate to the working directory
cd /storage/work/mqf5675/Masters/NLP/project/

# Run the Python script with optimized settings
echo "Starting translation model training..."
python3 transformer_en_de_ita_spa.py
echo "Job completed."

