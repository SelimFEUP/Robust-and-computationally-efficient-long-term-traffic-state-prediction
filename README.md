# Spatial-Temporal Robust Multi-step Traffic State Prediction
This repository presents a Long-Term Multi-Step Traffic State Prediction Using Attention Mechanism and Residual Temporal Convolutional Networks. The paper is published in the Neural Networks (https://doi.org/10.1016/j.neunet.2025.107897) with title 'Enhancing Intelligent Transportation Systems with a more Efficient Model for Long-Term Traffic Predictions based on an Attention Mechanism and a Residual Temporal Convolutional Network'.

# Datasets
Three datasets were used (i) PEMS-BAY (ii) PEMS, and (iii) METR-LA. The PEMS datset is uploaded in the data folder. Download PEMS-BAY.csv and METR-LA.csv from https://zenodo.org/records/5146275

# Note
Data augmentation was only applied to the PEMS dataset; PEMS-BAY and METR-LA do not require it.


# Usage
To train and evaluate the model, from the terminal run 
```bash 
python3 main.py
```

Also two robustness tests (i) Adversarial Perturbations and (ii) Random Sensors Dropout can be performed from the terminal using for an example
```bash 
python3 adversarial_perturbations_test.py
``` 

