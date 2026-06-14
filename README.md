# Rahulkumar-Celebal-intern

# WEEK-1

Machine Learning Foundations Assignment

This repository contains solutions for a comprehensive Machine Learning Foundations assignment covering Python programming, NumPy, Pandas, Linear Algebra, Statistics, and Probability Theory. The notebook demonstrates fundamental concepts required for Data Science and Machine Learning workflows.

📌 Topics Covered

🐍 Python Fundamentals
Data Types & Control Flow
Functions & Lambda Expressions
Data Structures (Lists, Sets, Dictionaries)
Exception Handling
🔢 NumPy
Array Creation & Reshaping
Indexing & Slicing
Vectorized Operations
Matrix Multiplication & Dot Product
🐼 Pandas
DataFrames & Series
Data Filtering
GroupBy Operations
Missing Value Handling
Data Aggregation
📐 Linear Algebra
Vector Representation
Matrix Operations
Eigenvalues & Eigenvectors
Singular Value Decomposition (SVD)
Dimensionality Reduction Concepts
📊 Statistics
Descriptive Statistics
Hypothesis Testing
Pearson Correlation
Error Metrics (MAE, MSE, RMSE, R², Adjusted R²)
Distribution Testing
Stationarity Testing (ADF Test)
Population Stability Index (PSI)
🎲 Probability Theory
Basic Probability
Conditional Probability
Joint Probability
Bayes Theorem
Normal Distribution
Binomial Distribution
Poisson Distribution
Central Limit Theorem (CLT)
🛠 Technologies Used
Python 3.x
NumPy
Pandas
Matplotlib
Seaborn
SciPy
Statsmodels

📂 Project Structure
Week1_ML_Foundations/
│
├── week1_assignment.ipynb
├── README.md
└── requirements.txt
📈 Key Concepts Implemented
Python

✔ Conditional Statements

✔ Exception Handling

✔ Functions & Lambda Expressions

✔ Dictionary, Set, List Comprehension

NumPy

✔ Array Manipulation

✔ Matrix Operations

✔ Boolean Indexing

✔ Dot Product

Pandas

✔ Data Cleaning

✔ Missing Value Imputation

✔ GroupBy Analysis

✔ Data Filtering

Linear Algebra

✔ Vector Norm

✔ Matrix Multiplication

✔ Eigen Decomposition

✔ Singular Value Decomposition (SVD)

Statistics

✔ Hypothesis Testing

✔ Correlation Analysis

✔ Distribution Testing

✔ Model Evaluation Metrics

Probability

✔ Bayes Theorem

✔ Probability Distributions

✔ Central Limit Theorem

✔ Conditional Probability

📊 Visualizations Included
Vector Visualization
Eigenvector Plot
Salary Distribution Histogram with KDE
Distribution Shift Visualization
Normal Distribution Curves
Binomial PMF
Poisson PMF
Central Limit Theorem Demonstration

🎯 Learning Outcomes

After completing this assignment, learners will be able to:

Write efficient Python code for data analysis.
Manipulate and analyze datasets using Pandas.
Perform numerical computations with NumPy.
Understand core Linear Algebra concepts used in ML.
Apply statistical methods for inference and hypothesis testing.
Interpret probability distributions and Bayesian reasoning.
Evaluate machine learning model performance using standard metrics.

🚀 How to Run

Install dependencies
pip install numpy pandas matplotlib seaborn scipy statsmodels
Open Jupyter Notebook
jupyter notebook
Run all cells from top to bottom.
✅ Assignment Status
Python Fundamentals ✔
NumPy ✔
Pandas ✔
Linear Algebra ✔
Statistics ✔
Probability Theory ✔
Visualizations ✔
Assertions Passed ✔
👨‍💻 Author

Rahul Kumar

# WEEK-2

README
🚗 Tesla Deliveries Prediction using Machine Learning
📌 Project Overview

This project builds a Machine Learning Regression model to predict Tesla vehicle deliveries using historical production, pricing, battery, charging infrastructure, and regional data from 2015–2025. The workflow covers data preprocessing, exploratory data analysis (EDA), model training, evaluation, feature importance analysis, and model persistence for future deployment.

🎯 Objectives
Analyze Tesla production and delivery trends.
Explore relationships between production, pricing, battery capacity, and deliveries.
Build a predictive model for Tesla estimated deliveries.
Identify key factors influencing deliveries.
Save the trained model for deployment and future predictions.
📊 Dataset Information

The dataset contains Tesla vehicle delivery and production records across multiple regions and models.

Features
Feature	Description
Year	Year of record
Month	Month of record
Region	Sales region
Model	Tesla model
Production_Units	Vehicles produced
Avg_Price_USD	Average vehicle price
Battery_Capacity_kWh	Battery capacity
Range_km	Vehicle range
CO2_Saved_tons	Environmental impact
Source_Type	Data source category
Charging_Stations	Number of charging stations
Target Variable
Estimated_Deliveries
🛠️ Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-Learn
Joblib
Kaggle Dataset
📈 Exploratory Data Analysis (EDA)
Dataset Inspection
Checked data types and structure.
Verified missing values.
Generated descriptive statistics.
Visualizations
Production vs Deliveries Scatter Plot

Analyzed the relationship between:

Production Units
Estimated Deliveries
Correlation Heatmap

Visualized correlations among numerical features to identify important relationships.

🔄 Data Preprocessing
Categorical Encoding

Applied Label Encoding to:

Region
Model
Source_Type
Feature Selection

Input Features (X):

Year
Month
Region
Model
Production_Units
Avg_Price_USD
Battery_Capacity_kWh
Range_km
CO2_Saved_tons
Source_Type
Charging_Stations

Target Variable (y):

Estimated_Deliveries
🤖 Machine Learning Model
Algorithm Used
Random Forest Regressor

Reasons for selecting Random Forest:

Handles non-linear relationships.
High predictive performance.
Robust against overfitting.
Works well with mixed feature types.
🔀 Train-Test Split
Training Data: 80%
Testing Data: 20%
Random State: 42
📉 Model Evaluation

The model was evaluated using:

Metric	Value
MAE	323.46
RMSE	409.50
R² Score	0.9888
Result

The model achieved an R² Score of 98.88%, indicating excellent predictive performance.

⭐ Feature Importance

Top influential features:

Feature	Importance
Production_Units	0.9908
CO2_Saved_tons	0.0023
Range_km	0.0014
Avg_Price_USD	0.0013
Charging_Stations	0.0012
Key Insight

Production Units is by far the most important factor affecting Tesla deliveries.

💾 Model Saving

The trained model was saved using Joblib:

import joblib

joblib.dump(model, "tesla_delivery_model.pkl")

Saved File:

tesla_delivery_model.pkl
📌 Key Insights
Tesla deliveries are highly dependent on production volume.
Strong positive correlation exists between production and deliveries.
Charging infrastructure and environmental impact contribute moderately.
Random Forest successfully captured complex relationships within the data.
The model demonstrates strong generalization on unseen data.
🚀 Future Improvements
Hyperparameter tuning using GridSearchCV.
Compare with XGBoost and LightGBM.
Deploy using Flask/FastAPI.
Build an interactive dashboard using Streamlit.
Incorporate real-world Tesla financial and market data.
📷 Project Outputs
Production vs Deliveries Scatter Plot
Correlation Heatmap
Feature Importance Visualization
Trained Regression Model
Delivery Predictions
✅ Conclusion

This project successfully developed an end-to-end Machine Learning pipeline for predicting Tesla vehicle deliveries. Using a Random Forest Regressor, the model achieved 98.88% accuracy (R² Score) and identified Production Units as the dominant factor influencing deliveries. The trained model can be further integrated into business intelligence or forecasting applications.

👨‍💻 Author

Rahul Kumar
Data Science & Machine Learning Enthusiast 🚀

# WEEK-3 

Country Intelligence System using Classification, Ensemble Learning & Clustering
Overview

This project presents an end-to-end Country Intelligence System that leverages Machine Learning and Unsupervised Learning techniques to analyze socioeconomic indicators of countries. The system combines Classification, Ensemble Learning, and Clustering algorithms to predict development status, identify patterns, and segment countries into meaningful groups.

Problem Statement

Understanding the socioeconomic conditions of countries is crucial for policy-making, economic planning, and international development. This project aims to analyze country-level data and generate actionable insights through predictive modeling and clustering techniques.

Dataset

The dataset contains various socioeconomic indicators such as:

Child Mortality
Exports
Health Expenditure
Imports
Income
Inflation
Life Expectancy
Total Fertility
GDP per Capita

Dataset Source: Country-data.csv

Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
XGBoost

Project Workflow

Data Collection
      ↓
Data Preprocessing
      ↓
Exploratory Data Analysis
      ↓
Feature Scaling
      ↓
Classification Models
(Logistic Regression,
 Random Forest,
 XGBoost)
      ↓
Model Evaluation
      ↓
Clustering
(K-Means, DBSCAN)
      ↓
PCA Visualization
      ↓
Insights & Recommendations
Exploratory Data Analysis (EDA)

Performed comprehensive EDA including:

Data quality assessment
Missing value analysis
Correlation Heatmap
Distribution Analysis
Feature Importance Analysis
Machine Learning Models
Classification
Logistic Regression
Baseline model used for development status prediction.

Random Forest

Ensemble learning model utilizing multiple decision trees for improved performance.

XGBoost

Gradient boosting algorithm known for high predictive accuracy and efficiency.

Clustering

K-Means Clustering

Grouped countries based on similar socioeconomic characteristics.

DBSCAN

Density-based clustering used for identifying hidden patterns and detecting outliers.

Dimensionality Reduction

Principal Component Analysis (PCA)

Used PCA to:

Reduce dimensionality
Visualize country clusters
Improve interpretability
Model Evaluation Metrics
Classification Metrics
Accuracy
Precision
Recall
F1-Score
ROC-AUC
Clustering Metrics
Silhouette Score
Cluster Analysis
Outlier Detection
Key Features

✔ Data Preprocessing Pipeline
✔ Correlation Analysis with Heatmap
✔ Feature Importance Visualization
✔ Ensemble Learning Models
✔ Country Segmentation using Clustering
✔ PCA-Based Cluster Visualization
✔ Outlier Detection using DBSCAN
✔ Actionable Data-Driven Insights

Results
Successfully implemented multiple classification models.
Compared baseline and ensemble learning performance.
Segmented countries into meaningful clusters.
Identified influential socioeconomic indicators.
Detected outlier countries using DBSCAN.
Generated insights to support economic and development analysis.

Project Structure

Country-Intelligence-System/
│
├── data/
│   └── Country-data.csv
│
├── notebooks/
│   └── Country_Intelligence_System.ipynb
│
├── images/
│   ├── heatmap.png
│   ├── feature_importance.png
│   └── pca_clusters.png
│
├── README.md
│
└── requirements.txt

Future Improvements

Deploy the project using FastAPI or Streamlit.
Integrate real-time economic datasets.
Build an interactive dashboard.
Automate model retraining and monitoring.

Author

Rahul Kumar
Data Science | Machine Learning | Artificial Intelligence


# WEEK-4

# 📘 CIFAR-10 Image Classification: ANN vs CNN

## Overview

This project compares **Artificial Neural Networks (ANN)** and **Convolutional Neural Networks (CNN)** on the CIFAR-10 image classification dataset. The goal is to understand how different deep learning architectures process image data and why CNNs are the preferred choice for computer vision tasks.

## Dataset

**CIFAR-10** consists of 60,000 color images (32×32 pixels) across 10 classes:

* Airplane

* Automobile

* Bird

* Cat

* Deer

* Dog

* Frog

* Horse

* Ship

* Truck

* Training Images: 50,000

* Test Images: 10,000

## Project Objectives

* Build and train an ANN model for image classification.
* Build and train a CNN model for image classification.
* Compare ANN and CNN performance.
* Analyze accuracy, loss curves, and generalization.
* Explore techniques such as Dropout, Batch Normalization, Data Augmentation, and Early Stopping.

## Model Architectures

### ANN

* Fully Connected Dense Layers
* ReLU Activation
* Dropout Regularization
* Softmax Output Layer

### CNN

* Convolutional Layers
* Max Pooling Layers
* Batch Normalization
* Dropout Layers
* Data Augmentation
* Early Stopping

## Results

| Model | Train Accuracy | Validation Accuracy | Test Accuracy |
| ----- | -------------- | ------------------- | ------------- |
| ANN   | 43.21%         | 45.54%              | 43.53%        |
| CNN   | 74.26%         | 69.71%              | 72.55%        |

## Key Findings

### ANN

* Learns from flattened pixel values.
* Ignores spatial relationships between pixels.
* Achieves moderate classification accuracy.
* More prone to overfitting on image datasets.

### CNN

* Preserves image structure and spatial information.
* Automatically extracts features such as edges, textures, and shapes.
* Learns hierarchical representations of images.
* Achieves significantly higher accuracy and better generalization.

## Training Improvements

### Dropout

Reduces overfitting by randomly disabling neurons during training.

### Batch Normalization

Stabilizes training and accelerates convergence.

### Data Augmentation

Generates transformed versions of images using:

* Rotation
* Horizontal Flip
* Zoom
* Width/Height Shifts

This increases dataset diversity and improves generalization.

### Early Stopping

Stops training when validation loss stops improving and restores the best model weights.

## Performance Analysis

### Accuracy

CNN achieved substantially higher training, validation, and test accuracy than ANN.

### Loss Curves

CNN showed faster convergence and lower validation loss, indicating more effective feature learning.

### Generalization

CNN generalized better to unseen images because convolution and pooling layers capture meaningful visual patterns.

## Conclusion

This project demonstrates that CNNs significantly outperform ANNs for image classification tasks. While ANNs can learn basic pixel-level patterns, they fail to preserve spatial information present in images. CNNs overcome this limitation through convolution, pooling, and hierarchical feature extraction.

The addition of Batch Normalization, Dropout, Data Augmentation, and Early Stopping further improved model robustness and generalization. These techniques represent industry-standard practices used in modern computer vision systems.

## Future Improvements

* Transfer Learning
* ResNet
* EfficientNet
* Vision Transformers (ViTs)
* Object Detection (YOLO)
* Image Segmentation

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* CIFAR-10 Dataset
