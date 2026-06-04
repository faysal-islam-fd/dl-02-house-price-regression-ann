# Ames Housing Price Prediction using Deep Learning

## Overview

This project builds a Deep Neural Network (DNN) model to predict house sale prices using the Ames Housing Dataset. The workflow includes data collection, exploratory data analysis (EDA), preprocessing, feature engineering, model training, and evaluation.

The goal is to accurately estimate property prices based on various housing characteristics using TensorFlow/Keras.

---

## Dataset

**Dataset:** Ames Housing Dataset

Source:

* Kaggle Ames Housing Dataset

The dataset contains detailed information about residential homes, including:

* Property size
* Construction quality
* Basement information
* Garage features
* Number of rooms
* Year built
* Sale price (target variable)

---

## Project Workflow

### 1. Data Collection

The dataset is downloaded using the Kaggle API and loaded into a Pandas DataFrame.

### 2. Exploratory Data Analysis (EDA)

Several analyses are performed:

#### Missing Value Analysis

* Missing value counts
* Missing value percentages
* Missing value heatmap visualization

#### Target Variable Analysis

* Original SalePrice distribution
* Log-transformed SalePrice distribution

#### Correlation Analysis

* Correlation matrix for numerical features
* Top features correlated with SalePrice

#### Feature Relationship Analysis

Visualization of important features against SalePrice:

* Overall Quality
* Ground Living Area
* Garage Capacity
* Basement Area
* Year Built
* Full Bathrooms

---

## Data Preprocessing

### Data Cleaning

Removed unnecessary identifier columns:

* Order
* PID

### Missing Value Handling

Numerical features:

* Filled using median values

Categorical features:

* Filled using mode values

### Encoding

Categorical variables are converted using:

```python
pd.get_dummies(drop_first=True)
```

### Target Transformation

To reduce skewness:

```python
y = np.log1p(SalePrice)
```

### Train-Test Split

```python
80% Training
20% Testing
```

### Feature Scaling

Standardization performed using:

```python
StandardScaler
```

---

## Deep Learning Model

### Architecture

The model is built using TensorFlow/Keras.

#### Components

* Dense Layers (ReLU activation)
* Batch Normalization
* Dropout Regularization
* Linear Output Layer

#### Example Structure

```text
Input Layer
     ↓
Dense Layer
     ↓
BatchNormalization
     ↓
Dropout
     ↓
Dense Layer
     ↓
BatchNormalization
     ↓
Dropout
     ↓
Output Layer (1 neuron)
```

---

## Training Configuration

### Optimizer

```python
Adam(learning_rate=0.001)
```

### Loss Function

```python
Mean Squared Error (MSE)
```

### Evaluation Metric

```python
Mean Absolute Error (MAE)
```

### Callbacks

#### Early Stopping

Stops training when validation performance no longer improves.

#### Reduce Learning Rate on Plateau

Automatically reduces learning rate when validation loss stagnates.

---

## Model Evaluation

The model is evaluated using:

### RMSE

Root Mean Squared Error

### MAE

Mean Absolute Error

### R² Score

Coefficient of Determination

---

## Visualizations

### Training Curves

* Training Loss vs Validation Loss
* Training MAE vs Validation MAE

### Residual Analysis

Residual plots are generated to inspect prediction errors and model behavior.

---

## Technologies Used

### Programming Language

* Python

### Libraries

* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* TensorFlow
* Keras
* SciPy

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/ames-house-price-prediction.git
cd ames-house-price-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Project

Open the Jupyter Notebook:

```bash
jupyter notebook
```

Run all cells sequentially to:

1. Download the dataset
2. Perform EDA
3. Preprocess data
4. Train the neural network
5. Evaluate performance
6. Generate visualizations

---

## Future Improvements

* Hyperparameter tuning
* Cross-validation
* Feature selection
* Ensemble learning
* XGBoost comparison
* Random Forest comparison
* Model deployment using Flask/FastAPI

---

## Author

**MD. Faysal Islam Fahad**

Web Application Developer & Machine Learning Enthusiast
