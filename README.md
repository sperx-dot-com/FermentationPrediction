# Conceptual Bioprocess Modelling Workflow in Python
This repository provides a conceptual and educational end-to-end workflow for processing, exploring and modeling fermentation and upstream bioprocess data using Python.
Although the dataset used here is synthetic, it reflects key characteristics of real E. coli bioprocess runs such as limited sample size, measurement noise, interacting variables and categorical process factors.

The goal of this project is to demonstrate how domain knowledge from bioprocess development can be combined with data-driven machine learning approaches to extract meaningful signals and make predictive models.

## Key Concepts

### 1. Synthetic Bioprocess Dataset
A mock dataset inspired by upstream E. coli fermentations is generated, including:

- induction temperature  
- pH  
- dissolved oxygen  
- feed rate  
- induction duration  
- OD at induction  
- strain type  
- tag type (e.g., CASPON, His, None)  
- titer (target variable)

This mimics typical structures of bioprocess development data.


### 2. Exploratory Data Analysis (EDA)

Basic descriptive statistics and visualizations to understand:

- distributions  
- correlations  
- noise patterns  
- possible biological effects  


### 3. Preprocessing Pipelines (scikit-learn)

A complete preprocessing workflow using:

- `StandardScaler` (numerical scaling)  
- `OneHotEncoder` (categorical encoding)  
- `ColumnTransformer` (combined preprocessing)  
- `Pipeline` (reproducible workflow)

This avoids data leakage and ensures consistent preprocessing for modelling.


### 4. Machine Learning Models

The notebook trains several regression models:

- **Linear Regression** (baseline)  
- **Random Forest Regressor**  
- **XGBoost Regressor**  
- **Small PyTorch feed-forward neural network**

Models are evaluated using:

- MSE  
- MAE  
- R²  

This illustrates how different model families perform on small, noisy bioprocess datasets.


## How to Interpret Model Metrics

Predictive modelling in bioprocess development requires metrics that are both technically meaningful and practically interpretable.

### R² (Coefficient of Determination)
Indicates how much variance in the target variable is explained by the model.

- 1.0 → perfect prediction  
- 0.0 → no better than mean prediction  
- <0 → worse than baseline  

Due to biological noise and small datasets, **R² values between 0.6 and 0.85 are often already strong** in bioprocess applications.

### MAE (Mean Absolute Error)
Average absolute deviation between predictions and true values.

- expressed in real units (e.g., g/L)
- easy to interpret  
- useful for practical decision-making

### MSE (Mean Squared Error)
Average squared deviation.

- penalizes large errors more strongly  
- often used as training loss  
- less intuitive due to squared unit  

### Summary Table

| Metric | Strength | When to Use |
|--------|----------|--------------|
| **R²** | Measures explained variance | Compare model fit |
| **MAE** | Interpretable in real units | Assess practical prediction accuracy |
| **MSE** | Penalizes large errors | Monitor robustness and outliers |


 

## Not Yet considered
- using different protein sequences to predict the titer based on the sequence
- for this, ESM embedding would be a possibility (to encode amino acid sequences)

