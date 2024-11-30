# SW-NN Model for Land Surface Temperature Retrieval

This repository contains the implementation of the SW-NN model, a robust framework for accurate land surface temperature (LST) retrieval. The SW-NN integrates the split-window approach with machine learning to enhance accuracy, particularly under conditions of uncertainty in emissivity and water vapor content.

## Features

- Implements a neural network-based enhancement of the split-window algorithm.
- Offers a model trained under the conditions of emissivity noise (LSE) with a standard deviation of 0.01, brightness temperature (BT) noise with a standard deviation of 0.05, and water vapor content (WVC) noise set at 10% of its value.
- Provides tools for direct evaluation and testing using the trained model.

## Requirements

Ensure you have the following installed:

- Python 3.8+
- PyTorch 1.9+
- NumPy
- pandas
- Matplotlib

Install the required Python packages using the following command:
```bash
pip install -r requirements.txt
```

## Files Overview

- **`model.py`**: Defines the SW-NN architecture.
- **`utils.py`**: Contains utility functions, including LST calculation based on coefficients.
- **`usage.ipynb`**: Jupyter Notebook demonstrating how to use the model for evaluation and predictions.
- **`data/`**: Directory for input test data.
- **`model/`**: Directory containing the model and normalization parameters .

## Usage

To evaluate the model and make predictions, refer to the provided Jupyter Notebook:

1. **Load the Model**:
   - The SW-NN model weights are available as `Sobrino_2000+NN_Emi0.01_WVC10%_BT0.05.pth`.

2. **Prepare Input Data**:
   - Input data should include brightness temperature (BT), emissivity (LSE), water vapor content (WVC), and viewing zenith angle (VZA).

3. **Run the Notebook**:
   - Open `usage.ipynb` in Jupyter Notebook and follow the provided steps to:
     - Load and normalize data.
     - Evaluate model accuracy against MOD11 and MOD21 products.
     - Predict LST using SW-NN and compare with in-situ measurements.

## Citation

