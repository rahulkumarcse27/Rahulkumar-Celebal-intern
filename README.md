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


  # WEEK-5

  # 📘 Deep Learning Text Generation Learning Project

## Text Generation using Vanilla RNN, LSTM, and GRU

This project demonstrates how Deep Learning sequence models learn language patterns, grammar, contextual relationships, and next-word prediction for text generation.

The notebook is designed for students and beginners to understand the differences between **Vanilla RNN**, **LSTM**, and **GRU** architectures through hands-on experiments.

---

## 🎯 Project Objective

Design and implement Deep Learning models capable of learning the structure, grammar, and contextual dependencies of text to generate meaningful sequences.

Models Implemented:

* Vanilla RNN
* LSTM (Long Short-Term Memory)
* GRU (Gated Recurrent Unit)

The project compares:

* Training Loss
* Generated Text Quality
* Memory Handling
* Long-Term Dependency Learning
* Computational Efficiency

---

## 🧠 Learning Outcomes

By completing this project, students will learn:

* Text preprocessing and tokenization
* Sequence generation for next-word prediction
* Word embeddings
* Recurrent Neural Networks
* LSTM architecture and gating mechanisms
* GRU architecture and memory optimization
* Text generation using trained sequence models
* Comparison of sequence modeling architectures

---

## 📂 Project Workflow

```text
Text Corpus
     ↓
Tokenization
     ↓
Sequence Generation
     ↓
Padding
     ↓
Input Features (X)
Target Labels (y)
     ↓
Embedding Layer
     ↓
RNN / LSTM / GRU
     ↓
Softmax Output
     ↓
Next Word Prediction
     ↓
Text Generation
```

---

## ⚙️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib

---

## 📥 Datasets Used

A small built-in corpus is used for quick experimentation.

Students can replace it with:

* Shakespeare Text
* Song Lyrics
* Chatbot Conversations
* Story Paragraphs
* Custom PDF Extracted Text
* Research Articles
* News Articles

---

## 🔤 Text Preprocessing

The project performs:

### Tokenization

Converts words into integer tokens.

Example:

```python
deep learning is transforming artificial intelligence
```

↓

```text
[1, 2, 3, 4, 5]
```

### Sequence Generation

Creates n-gram sequences:

```text
deep learning
deep learning is
deep learning is transforming
```

### Padding

Ensures all sequences have equal length before training.

---

## 🧠 Model Architectures

### 1. Vanilla RNN

```python
Embedding
↓
SimpleRNN(128)
↓
Dense(Softmax)
```

#### Advantages

* Simple architecture
* Fast training
* Easy to understand

#### Limitations

* Suffers from vanishing gradients
* Poor long-term memory

---

### 2. LSTM

```python
Embedding
↓
LSTM(128)
↓
Dense(Softmax)
```

#### Features

* Input Gate
* Forget Gate
* Output Gate

#### Advantages

* Handles long-term dependencies
* Better contextual understanding
* Improved text generation quality

---

### 3. GRU

```python
Embedding
↓
GRU(128)
↓
Dense(Softmax)
```

#### Features

* Update Gate
* Reset Gate

#### Advantages

* Faster than LSTM
* Fewer parameters
* Similar performance to LSTM

---

## 📊 Model Comparison

| Feature                | RNN   | LSTM      | GRU       |
| ---------------------- | ----- | --------- | --------- |
| Memory Handling        | Poor  | Excellent | Very Good |
| Long-Term Dependencies | Weak  | Strong    | Strong    |
| Training Speed         | Fast  | Slowest   | Faster    |
| Complexity             | Low   | High      | Medium    |
| Text Quality           | Basic | Best      | Near LSTM |

---

## 🧪 Experiments Performed

### Experiment 1: AI Corpus

```text
machine learning is a subset of artificial intelligence
deep learning uses neural networks with multiple layers
data quality plays an important role in model performance
```

Sample Output:

```text
RNN  : machine learning is a subset of artificial intelligence ...
LSTM : machine learning is a subset of artificial intelligence ...
GRU  : machine learning is a subset of artificial intelligence ...
```

---

### Experiment 2: Shakespeare Corpus

```text
to be or not to be that is the question
whether tis nobler in the mind to suffer
the slings and arrows of outrageous fortune
```

Sample Output:

```text
RNN  : to be or not to be that is the question ...
LSTM : to be or not to be that is the question ...
GRU  : to be or not to be that is the question ...
```

---

### Experiment 3: Song Lyrics

```text
here comes the sun and i say
its all right little darling
the smiles returning to the faces
```

---

### Experiment 4: Chatbot Conversations

```text
hello how are you
i am doing well thank you
what can you help me with
i can answer questions and provide information
```

---

### Experiment 5: Story Generation

```text
once upon a time there was a brave knight
the knight lived in a beautiful kingdom
he protected the people from danger
everyone admired his courage and kindness
```

---

## 📉 Training Loss Analysis

Training loss curves are plotted for:

* RNN
* LSTM
* GRU

The graphs help visualize:

* Convergence speed
* Learning stability
* Model performance differences

---

## ✍️ Text Generation

After training, each model predicts the next word repeatedly to generate text.

Example:

```python
generate_text(model, "machine learning", 10)
```

Output:

```text
machine learning is a subset of artificial intelligence ...
```

---

## 🎓 Student Tasks

### Beginner

* Replace corpus with your own text
* Increase epochs from 100 → 200
* Increase embedding dimension
* Change hidden units from 64 → 128
* Generate longer text sequences

### Intermediate

* Add Dropout
* Add Bidirectional LSTM
* Use larger datasets
* Experiment with temperature sampling

### Advanced

* Attention Mechanisms
* Transformer Models
* GPT-style Text Generation
* Character-Level Text Generation

---

## 🚀 Future Improvements

* Beam Search Decoding
* Transformer Architecture
* GPT Implementation
* Pretrained Word Embeddings
* Large Corpus Training
* PDF Text Extraction Pipeline
* Interactive Web Interface

---

## 🏆 Conclusion

This project demonstrates the fundamentals of sequence modeling and text generation using Deep Learning.

Key Findings:

* Vanilla RNN learns basic language patterns but struggles with long-term memory.
* LSTM produces the most coherent and context-aware text.
* GRU achieves performance close to LSTM while training faster.
* Dataset quality and size significantly influence generated text quality.

This project serves as an excellent foundation for understanding modern NLP systems, language models, and generative AI architectures.


# WEEK-6

# 🖼️ Image Denoising Using Autoencoder on MNIST

## 📌 Overview

This project implements a **Convolutional Autoencoder** using **TensorFlow/Keras** to remove noise from handwritten digit images in the MNIST dataset. The model learns to reconstruct clean images from noisy inputs, demonstrating the power of deep learning for image restoration.

---

## 🎯 Aim

To build a deep learning model using an autoencoder on the MNIST dataset for removing noise from images and reconstructing clean images.

---

## ✨ Features

* Load and preprocess the MNIST dataset
* Normalize grayscale images
* Add Gaussian noise to images
* Build a Convolutional Autoencoder
* Train the model using noisy images as input
* Reconstruct clean images from noisy inputs
* Visualize original, noisy, and denoised images
* Plot training and validation loss

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib

---

## 📂 Project Structure

```
Image-Denoising-Autoencoder/
│
├── train.py
├── requirements.txt
├── README.md
└── autoencoder.keras
```

---

## 📊 Dataset

**Dataset:** MNIST

* 60,000 training images
* 10,000 testing images
* Image size: 28 × 28 pixels
* Grayscale handwritten digits (0–9)

---

## ⚙️ Methodology

### Step 1: Import Libraries

Import TensorFlow, NumPy, Matplotlib, and the required Keras modules.

### Step 2: Load MNIST Dataset

Load the handwritten digit dataset using `mnist.load_data()`.

### Step 3: Data Preprocessing

* Normalize pixel values between 0 and 1.
* Reshape images to `(28, 28, 1)`.

### Step 4: Add Gaussian Noise

Random Gaussian noise is added to create noisy input images while keeping the original images as targets.

### Step 5: Build the Autoencoder

#### Encoder

* Conv2D (32 filters)
* MaxPooling2D
* Conv2D (16 filters)
* MaxPooling2D

#### Decoder

* Conv2D (16 filters)
* UpSampling2D
* Conv2D (32 filters)
* UpSampling2D
* Conv2D (1 filter, Sigmoid activation)

### Step 6: Compile the Model

* Optimizer: Adam
* Loss Function: Binary Crossentropy

### Step 7: Train the Model

The model is trained using:

* Input → Noisy Images
* Target → Original Clean Images

Training Parameters:

* Epochs: 10
* Batch Size: 128

### Step 8: Image Reconstruction

The trained autoencoder predicts clean images from noisy test images.

### Step 9: Visualization

The following images are displayed:

* Original Images
* Noisy Images
* Denoised Images

### Step 10: Performance Evaluation

Training and validation loss are plotted to observe model convergence.

---

## 🧠 Model Architecture

```
Input Image (28×28×1)
          │
          ▼
Conv2D (32)
          │
MaxPooling
          │
Conv2D (16)
          │
MaxPooling
          │
Encoded Representation
          │
Conv2D (16)
          │
UpSampling
          │
Conv2D (32)
          │
UpSampling
          │
Conv2D (1)
          │
          ▼
Denoised Image
```

---

## 📈 Results

* Successfully removed Gaussian noise from handwritten digit images.
* Preserved important digit features during reconstruction.
* Training and validation loss decreased consistently over epochs, indicating effective learning.

---

## 📷 Output

The project displays:

* Original Image
* Noisy Image
* Reconstructed (Denoised) Image

It also generates a graph showing:

* Training Loss
* Validation Loss

---

## 🚀 How to Run

### Clone the Repository

```bash
git clone https://github.com/your-username/Image-Denoising-Autoencoder.git
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Project

```bash
python train.py
```

---

## 📦 Requirements

```
tensorflow
numpy
matplotlib
```

---

## 🎓 Learning Outcomes

After completing this project, you will understand:

* Autoencoders
* Convolutional Neural Networks (CNNs)
* Image Denoising
* Image Reconstruction
* Deep Learning Model Training
* TensorFlow/Keras Implementation
* Computer Vision Fundamentals

---

## 🔮 Future Improvements

* Train on color image datasets (CIFAR-10)
* Use Denoising U-Net architecture
* Apply the model to medical images
* Experiment with different noise levels
* Compare with Variational Autoencoders (VAE)
* Evaluate using PSNR and SSIM metrics

---

## 📌 Conclusion

A deep learning autoencoder model was successfully developed to remove Gaussian noise from MNIST handwritten digit images. The model learned to reconstruct clean images from noisy inputs while preserving essential digit features. The decreasing training and validation losses demonstrate effective learning, making autoencoders a powerful approach for image denoising and reconstruction tasks.

---

## 👨‍💻 Author

**Rahul Kumar**

B.Tech CSE (AI & ML)


# WEEK - 8

# 🤖 Single Agent Smart Assistant

## Project Overview

The **Single Agent Smart Assistant** is a Python-based intelligent assistant that understands user queries, identifies their intent, routes them to the appropriate tool, and returns a structured JSON response.

The project demonstrates the fundamentals of AI agent design, conditional routing, tool integration, logging, and structured output.

---

## Features

* Intent Detection
* Conditional Routing
* Calculator Tool
* Keyword Extraction Tool
* General Query Handling
* Structured JSON Response
* Request Counter
* Date & Time (Timestamp)
* Logging (`agent.log`)
* Error Handling

---

## Project Structure

```
Single-Agent-Smart-Assistant/
│
├── main.py
├── agent.py
├── tools.py
├── agent.log
├── README.md
```

---

## Tools

### 1. Calculator Tool

Performs mathematical calculations.

Example:

```
Calculate 20 + 5
```

---

### 2. Keyword Extraction Tool

Extracts important keywords from a sentence.

Example:

```
Extract keywords from Artificial Intelligence is transforming industries
```

---

### 3. General Response

Handles all other user queries.

Example:

```
What is Machine Learning?
```

---

## Technologies Used

* Python 3
* datetime
* logging
* re

---

## Sample JSON Output

```json
{
    "request_id": 1,
    "type": "calculator",
    "tool": "Calculator",
    "query": "Calculate 20 + 5",
    "result": "25",
    "timestamp": "2026-07-12 15:20:10",
    "status": "success"
}
```

---

## Bonus Features

* Improved Routing
* Logging
* Request Counter
* Timestamp
* Error Handling
* Modular Tool Design

---

## Future Enhancements

* Weather Tool
* Translator Tool
* Sentiment Analysis Tool
* Currency Converter
* Wikipedia Search
* Voice Assistant
* GUI using Streamlit or Flask

---

## Author

**Rahul Kumar**

B.Tech CSE (AI & ML)

Single Agent Smart Assistant Project



