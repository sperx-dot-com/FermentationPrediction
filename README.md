# Conceptual Bioprocess Modelling Workflow in Python
This repository provides a conceptual and educational end-to-end workflow for processing, exploring and modeling fermentation and upstream bioprocess data using Python.
Although the dataset used here is synthetic, it reflects key characteristics of real E. coli bioprocess runs such as limited sample size, measurement noise, interacting variables and categorical process factors.

The goal of this project is to demonstrate how domain knowledge from bioprocess development can be combined with data-driven machine learning approaches to extract meaningful signals and make predictive models.

## 🧪 Key Concepts

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



### 5. Feature Importance and Interpretation

Feature importances are extracted from the Random Forest model to identify:

- influential process variables  
- effects of strain and tag type  
- relationships between parameters and titer  
- connections to bioprocess domain knowledge  


## 🎯 Purpose of This Repository

This project serves as a demonstration and learning tool for professionals at the interface of:

- bioprocess development  
- protein and strain engineering  
- analytical data processing (e.g., LC–MS)  
- machine learning with Python  

It highlights:

- how to design a bioprocess modelling workflow  
- how to integrate biological expertise into modelling  
- how to handle small or noisy datasets  
- how predictive models can support development decisions


## Not Yet considered
- using different protein sequences to predict the titer based on the sequence
- for this, ESM embedding would be a possibility (to encode amino acid sequences)

