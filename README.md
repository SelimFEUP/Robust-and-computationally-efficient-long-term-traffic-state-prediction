# Spatial-Temporal Robust Multi-step Traffic State Prediction
This repository presents a ''Long-Term Multi-Intersection Traffic State Prediction Using Attention Mechanism and Residual Temporal Convolutional Networks''.

# Datasets
Three datasets were used (i) PEMS-BAY (ii) PEMS, and (iii) METR-LA. The PEMS datset is uploaded in the data folder. Download PEMS-BAY.csv and METR-LA.csv from https://zenodo.org/records/5146275

# Note
Data augmentation was only applied to the PEMS dataset; PEMS-BAY and METR-LA do not require it.


# Usage
From the terminal run 'python3 main.py'.
Also two robustness tests (i) Adversarial Perturbations and (ii) Random Sensors Dropout can be performed from the terminal using for an example 'python3 ./Robustness_test/Adversarial_Perturbations_Test.py' 


