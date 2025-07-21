# wsBNN: Weight-sharing Bayesian Neural Network for Feature Selection

This repository contains the implementation of the **Weight-sharing Bayesian Neural Network (wsBNN)** model for feature selection. The code is designed to work with various datasets. Follow the steps below to set up the environment and run the model.

## Installation

1. Clone the repository
   ```bash
   git clone https://github.com/AkankshaMishra/wsBNN.git
   cd wsBNN
   
2. Create and activate a Conda environment
   ```bash
   conda create -n fs python=3.9
   conda activate fs

3. Install the required dependencies
   ```bash
   pip install -r requirements.txt
   
4. Navigate to the task folder
   Replace <task> with the name of the task you want to perform (classification or regression).
   ```bash
   cd <task>

5. Navigate to the dataset folder
   Replace <dataset-folder> with the name of the specific dataset you want to use.
   ```bash
   cd <task>/<dataset-folder>

7. Run the fsbnn.py script
   You can optionally modify hyperparameters or experiment settings directly in the file.
   ```bash
   python test_fsbnn_sm.py
  
Note: The script assumes the dataset files are already present in the respective dataset folders. Logs and results will be saved in the folder where the script is executed.
