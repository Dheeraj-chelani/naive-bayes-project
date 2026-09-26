# Naive Bayes Algorithms From Scratch

> A machine learning project implementing the major variants of **Naive Bayes classification from scratch using Python and NumPy**.

---

##  About the Project

This project implements the major Naive Bayes variants from scratch without using ready-made classifiers.

The goal is to understand the complete working of Naive Bayes, including:

- Bayes' Theorem
- Conditional probability
- Class priors and likelihoods
- Conditional independence assumption
- Smoothing
- Prediction
- Model evaluation
- Mathematical foundations
- Comparison with Scikit-learn implementations

Each completed implementation is directly compared with the corresponding **Scikit-learn** version to validate predictions, metrics, and performance.

---

##  Objectives

- Understand the mathematics behind Naive Bayes.
- Implement each major variant from scratch.
- Perform EDA and data preprocessing.
- Use the appropriate feature representation for each variant.
- Calculate class priors and likelihoods manually.
- Apply smoothing techniques.
- Evaluate models using standard classification metrics.
- Compare implementations against Scikit-learn.
- Understand hyperparameters, complexity, applications, advantages, and limitations.

---

##  Project Overview

| Algorithm | Dataset | Status |
|---|---|---|
| Gaussian Naive Bayes | Wine Quality Red | ✅ Completed |
| Multinomial Naive Bayes | SMS Spam Collection | ✅ Completed |
| Bernoulli Naive Bayes | SMS Spam Collection | ✅ Completed |

---

#  Naive Bayes Concept

Naive Bayes is a probabilistic classification algorithm based on **Bayes' Theorem**:


P(A | B) = [P(B | A) × P(A)] / P(B)


Where:

| Term | Meaning |
|---|---|
| `P(C\|X)` | Posterior probability |
| `P(X\|C)` | Likelihood |
| `P(C)` | Prior probability |
| `P(X)` | Evidence |

### Conditional Independence Assumption

Naive Bayes assumes that features are **conditionally independent given the class**:

P(x_1,...,x_d | y) = Π P(x_i | y)

---

##  Naive Bayes Variants

| Algorithm | Feature Type | Representation | Main Use |
|---|---|---|---|
| Gaussian NB | Continuous numerical | Mean & variance | Numerical data |
| Multinomial NB | Counts / frequencies | Word counts | Text classification |
| Bernoulli NB | Binary / Boolean | Presence or absence | Binary features & text |

---

# 1. Gaussian Naive Bayes

**Status:** ✅ Completed

### Dataset

**Wine Quality Red**

- 11 physicochemical features
- Quality `>= 6` → Good
- Quality `< 6` → Bad

### Workflow

```text
Data Cleaning
     ↓
EDA
     ↓
Train-Test Split
     ↓
Class Priors
     ↓
Class-wise Mean & Variance
     ↓
Gaussian PDF
     ↓
Log Probability
     ↓
Prediction
     ↓
Evaluation
     ↓
Scikit-learn Comparison
```

### Hyperparameter

`var_smoothing = 1e-9`

This adds a small value to variances to avoid numerical issues with near-zero variance.

### Results

| Metric | From Scratch | Scikit-learn |
|---|---:|---:|
| Accuracy | 72.43% | 72.43% |
| Precision | 76.34% | 76.34% |
| Recall | 69.44% | 69.44% |
| F1 Score | 72.73% | 72.73% |

**Prediction Agreement:** 100% with Scikit-learn.

---

# 2. Multinomial Naive Bayes

**Status:** ✅ Completed

### Dataset

**SMS Spam Collection**

- Ham = `0`
- Spam = `1`
- Designed for text classification using word frequencies.

### Workflow

```text
Dataset Preparation
     ↓
EDA
     ↓
Text Preprocessing
     ↓
Vocabulary & Word-Index Mapping
     ↓
Word-Count Vectorization
     ↓
Class Priors
     ↓
Class-wise Word Counts
     ↓
Laplace Smoothing
     ↓
Log Probability
     ↓
Prediction
     ↓
Evaluation
     ↓
Scikit-learn Comparison
```

### Feature Representation

Each message is converted into a **word-count vector**.

Example:

```text
"free free prize"
```

becomes:

```text
free  → 2
prize → 1
```

### Hyperparameter

`alpha = 1.0`

Laplace smoothing prevents zero probability for unseen words.

$$
P(w\vert{}c)=
\frac{count(w,c)+\alpha}
{total\ words\ in\ c+\alpha V}
$$

### Results

| Metric | From Scratch | Scikit-learn |
|---|---:|---:|
| Accuracy | 97.97% | 97.97% |
| Precision | 94.35% | 94.35% |
| Recall | 89.31% | 89.31% |
| F1 Score | 91.76% | 91.76% |

**Metrics:** Match exactly between the two implementations.

---

# 3. Bernoulli Naive Bayes

**Status:** ✅ Completed

### Dataset

**SMS Spam Collection**

Bernoulli Naive Bayes uses **binary features**, representing whether a word/feature is present (`1`) or absent (`0`) rather than using its count.

### Workflow

```text
Dataset Preparation
     ↓
EDA
     ↓
Text Preprocessing
     ↓
Binary Feature Representation
     ↓
Class Priors
     ↓
Feature Probabilities
     ↓
Laplace Smoothing
     ↓
Training
     ↓
Prediction
     ↓
Evaluation
     ↓
Confusion Matrix
     ↓
Scikit-learn Comparison
```

### Hyperparameter

`alpha = 1.0`

Laplace smoothing is used to avoid zero probabilities.

### Results

| Metric | From Scratch | Scikit-learn |
|---|---:|---:|
| Accuracy | 97.97% | 97.97% |
| Precision | 99.11% | 99.11% |
| Recall | 84.73% | 84.73% |
| F1 Score | 91.36% | 91.36% |

**Prediction Agreement:** 100% with the Scikit-learn equivalent.

---

#  Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- TP / TN / FP / FN

### F1 Score

$$
F1 = 2 \times
\frac{Precision \times Recall}
{Precision + Recall}
$$

---

##  From-Scratch vs Scikit-learn

Each manual implementation is compared with the corresponding Scikit-learn implementation using:

- Predictions
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Prediction agreement
- Computational complexity

Scikit-learn is used as the **reference/validation baseline**.

---

#  Applications

| Variant | Typical Applications |
|---|---|
| Gaussian NB | Medical, sensor/measurement, financial, quality/scientific data classification |
| Multinomial NB | Spam detection (SMS/email), sentiment analysis, news/document/text categorization |
| Bernoulli NB | Text classification with binary features, spam & document classification, presence/absence problems |

---

#  Advantages

- Simple to understand
- Fast to train and predict
- Works well on high-dimensional data
- Requires relatively little training data
- Effective for text classification
- Easy to implement
- Good introduction to probabilistic machine learning

---

#  Limitations

- The conditional-independence assumption often does not hold in practice.
- Performance can drop when features are strongly dependent.
- Probability estimates can be poorly calibrated.
- Each Naive Bayes variant requires the appropriate feature representation.
- Text models can be sensitive to vocabulary and preprocessing choices.

---

#  Technologies Used

| Technology | Purpose |
|---|---|
| **Python** | Programming language |
| **NumPy** | From-scratch mathematical implementation |
| **Pandas** | Data handling |
| **Matplotlib** | Visualization |
| **Seaborn** | Visualization |
| **Scikit-learn** | Reference and comparison |
| **Jupyter Notebook** | Development and experimentation |

---

#  Project Structure

```text
naive-bayes-project/
│
├── dataset/
│   ├── winequality-red.csv
│   └── SMSSpamCollection
│
├── notebooks/
│   ├── gaussian_nb_project.ipynb
│   ├── multinomial_nb_project.ipynb
│   └── bernoulli_nb_project.ipynb
│
├── .gitignore
└── README.md
```

---

#  Project Status

| Model | Completion |
|---|---:|
| Gaussian Naive Bayes | ✅ 100% |
| Multinomial Naive Bayes | ✅ 100% |
| Bernoulli Naive Bayes | ✅ 100% |

---

#  Conclusion

This project implements and explains major Naive Bayes variants from scratch using **Python and NumPy**.

- **Gaussian Naive Bayes** was successfully implemented for the Wine Quality Red dataset.
- **Multinomial Naive Bayes** and **Bernoulli Naive Bayes** achieved approximately **98% accuracy** on the SMS Spam Collection dataset.
- The from-scratch implementations produced results matching their corresponding Scikit-learn implementations.
- The project covers the mathematical foundations, assumptions, implementation details, feature representations, evaluation metrics, complexity, hyperparameters, applications, advantages, and limitations of each variant.

Overall, the project provides a practical understanding of how Naive Bayes classifiers work internally rather than relying only on library implementations.
