# Gaussian Naive Bayes From Scratch

## 📌 Project Overview

This project focuses on understanding and implementing the Gaussian Naive Bayes classification algorithm from scratch using Python and NumPy.

The main objective of this project is to understand the mathematical intuition and working of Gaussian Naive Bayes instead of simply using a ready-made machine learning implementation.

The from-scratch implementation is also compared with the Scikit-learn implementation to verify the results.

---

## 🎯 Objectives

- Understand Bayes' theorem and the Naive Bayes assumption.
- Understand how Gaussian Naive Bayes works with continuous numerical data.
- Perform Exploratory Data Analysis (EDA).
- Handle data preprocessing and duplicate records.
- Implement Gaussian Naive Bayes from scratch using NumPy.
- Understand mean, variance, prior probability, and likelihood calculations.
- Evaluate the model using different performance metrics.
- Compare the from-scratch implementation with Scikit-learn.
- Analyze model predictions and errors.

---

## 📊 Dataset

### Wine Quality Dataset

The Wine Quality dataset is used for this project.

The dataset contains physicochemical properties of red wine such as:

- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol

The original target variable is `quality`, which contains wine quality scores.

For binary classification, the target is converted into two classes:

- **Good** → Quality >= 6
- **Bad** → Quality < 6

### Dataset Source

UCI Machine Learning Repository

---

## 🔬 Exploratory Data Analysis

The following EDA steps are performed:

- Dataset structure analysis
- Missing value analysis
- Duplicate detection and removal
- Target class distribution
- Numerical feature analysis
- Outlier analysis
- Correlation analysis
- Feature distribution analysis
- Comparison of features between Good and Bad wines

Several visualizations are also used to understand the dataset and identify useful patterns.

---

## 🧠 Gaussian Naive Bayes

Gaussian Naive Bayes is a classification algorithm that is suitable for continuous numerical features.

It assumes that the features follow an approximately Gaussian (normal) distribution within each class.

The Gaussian probability density function used is:

P(x|y) = 1 / √(2πσ²) × exp(-(x-μ)² / (2σ²))

Where:

- `μ` = Mean of the feature
- `σ²` = Variance of the feature
- `x` = Feature value

The algorithm calculates:

1. Class prior probability
2. Mean of each feature for each class
3. Variance of each feature for each class
4. Feature likelihoods
5. Overall class score
6. Final predicted class

---

## ⚙️ From-Scratch Implementation

Gaussian Naive Bayes is implemented from scratch using:

- Python
- NumPy

No ready-made Naive Bayes classifier is used in the from-scratch implementation.

The implementation includes:

- `fit()` method for calculating model parameters
- Gaussian probability density function
- Log probability calculation
- `predict()` method for making predictions

---

## 📈 Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### From-Scratch Results

| Metric | Score |
|---|---:|
| Accuracy | 72.43% |
| Precision | 76.34% |
| Recall | 69.44% |
| F1 Score | 72.73% |

---

## 🔄 From-Scratch vs Scikit-learn

The from-scratch implementation is compared with the Gaussian Naive Bayes implementation provided by Scikit-learn.

The comparison includes:

- Accuracy
- Precision
- Recall
- F1 Score
- Prediction agreement

The from-scratch implementation produced the same predictions as the Scikit-learn implementation on the test dataset.

**Prediction Agreement: 100%**

This confirms that the core implementation is working consistently with the standard library implementation for this dataset and setup.

---

## 📁 Project Structure

```text
naive-bayes-project/
│
├── dataset/
│   └── winequality-red.csv
│
├── notebooks/
│   └── naive_bayes_project.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore