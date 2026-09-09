# Naive Bayes Algorithms From Scratch

> A machine learning project implementing the major variants of **Naive Bayes classification from scratch using Python and NumPy**.

---

## About the Project

This project implements Naive Bayes variants from scratch (no ready-made classifiers) to understand Bayes' Theorem, conditional probability, priors/likelihoods, the independence assumption, smoothing, prediction, and evaluation.

| Algorithm | Dataset | Status |
|---|---|---|
| Gaussian Naive Bayes | Wine Quality Red | Completed |
| Multinomial Naive Bayes | SMS Spam Collection | Completed |
| Bernoulli Naive Bayes | Text Classification | In Progress (50%) |

Each completed implementation is compared with the corresponding **Scikit-learn** version.

### Objectives
Understand the math behind Naive Bayes; implement each variant from scratch; perform EDA and preprocessing; use the correct feature representation per variant; evaluate with standard classification metrics; compare against Scikit-learn; understand hyperparameters, complexity, applications, advantages, and limitations of each variant.

---

## Naive Bayes Concept

Naive Bayes is a probabilistic classifier based on **Bayes' Theorem**:

\[
P(C|X)=\frac{P(X|C)P(C)}{P(X)}
\]

- `P(C|X)` → Posterior, `P(X|C)` → Likelihood, `P(C)` → Prior, `P(X)` → Evidence

It assumes features are **conditionally independent given the class**:

\[
P(X|C)=\prod_i P(x_i|C)
\]

### Variant Comparison

| Algorithm | Feature Type | Representation | Main Use |
|---|---|---|---|
| Gaussian NB | Continuous numerical | Mean & variance | Numerical data |
| Multinomial NB | Counts / frequencies | Word counts | Text classification |
| Bernoulli NB | Binary / Boolean | Presence or absence | Binary features & text |

---

## 1. Gaussian Naive Bayes — Completed

Used for **continuous numerical features**, on the **Wine Quality Red** dataset (11 physicochemical features; quality ≥ 6 → Good, < 6 → Bad).

**Steps:** data cleaning → EDA → train-test split → class priors → class-wise mean/variance → Gaussian PDF → log probability → prediction → evaluation → Scikit-learn comparison.

**Hyperparameter:** `var_smoothing = 1e-9` (adds a small value to variances to avoid numerical issues with near-zero variance).

| Metric | From Scratch | Scikit-learn |
|---|---:|---:|
| Accuracy | 72.43% | 72.43% |
| Precision | 76.34% | 76.34% |
| Recall | 69.44% | 69.44% |
| F1 Score | 72.73% | 72.73% |

Identical predictions on all 272 test samples — **100% agreement** with Scikit-learn.

---

## 2. Multinomial Naive Bayes — Completed

Used for **text classification** based on word frequency, on the **SMS Spam Collection** dataset (Ham → 0, Spam → 1).

**Steps:** data cleaning → EDA → text preprocessing (lowercasing, cleanup) → vocabulary & word-index mapping → word-count vectorization → class priors → class-wise word counts → Laplace smoothing → log probability → prediction → evaluation → Scikit-learn comparison.

Each message becomes a word-count vector, e.g. `"free free prize"` → `free: 2, prize: 1`.

**Hyperparameter:** `alpha = 1.0` (Laplace smoothing, prevents zero probability for unseen words):

\[
P(w|c)=\frac{count(w,c)+\alpha}{total\ words\ in\ c+\alpha V}
\]

| Metric | From Scratch | Scikit-learn |
|---|---:|---:|
| Accuracy | 97.97% | 97.97% |
| Precision | 94.35% | 94.35% |
| Recall | 89.31% | 89.31% |
| F1 Score | 91.76% | 91.76% |

Metrics match exactly between the two implementations.

---

## 3. Bernoulli Naive Bayes — In Progress

Uses **binary features**: whether a word/feature is present (`1`) or absent (`0`), rather than its count. Will also use the SMS Spam Collection dataset.

**Planned steps:** dataset prep → EDA → text preprocessing → binary feature representation → class priors → feature probabilities → Laplace smoothing → training → prediction → evaluation → confusion matrix → Scikit-learn comparison → complexity/hyperparameter analysis.

Results and hyperparameter details will be added once complete.

---

## Model Evaluation

Models are evaluated using **Accuracy, Precision, Recall, F1 Score, and the Confusion Matrix** (TP/TN/FP/FN).

\[
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
\]

### From-Scratch vs Scikit-learn
Each variant's manual implementation is compared to Scikit-learn's on predictions, all metrics above, confusion matrix, prediction agreement, and computational complexity — using Scikit-learn as the reference/validation baseline.

---

###  Project Workflow 
The overall workflow used in the project is: 
Dataset
|
v
Data Cleaning
|
v
EDA
|
v
Data Preprocessing
|
v
Feature Representation
|
v
Train-Test Split
|
v
Naive Bayes From Scratch
|
v
Prediction
|
v
Evaluation
|
v
Scikit-learn Model
|
v
Comparison

## Applications, Advantages & Limitations

| Variant | Typical Applications |
|---|---|
| Gaussian NB | Medical, sensor/measurement, financial, and quality/scientific data classification |
| Multinomial NB | Spam detection (SMS/email), sentiment analysis, news/document/text categorization |
| Bernoulli NB | Text classification with binary features, spam & document classification, presence/absence problems |

**Advantages:** simple to understand, fast to train/predict, works well on high-dimensional data, needs relatively little training data, effective for text classification, easy to implement, good intro to probabilistic ML.

**Limitations:** the conditional-independence assumption often doesn't hold in practice; performance drops with strongly dependent features; probability estimates can be poorly calibrated; each variant needs the right feature representation; text models are sensitive to vocabulary/preprocessing choices.

---

## Technologies Used

Python, NumPy (from-scratch implementation), Pandas (data handling), Matplotlib & Seaborn (visualization), Scikit-learn (reference/comparison), Jupyter Notebook, VS Code.

---

## Project Structure

```text
naive-bayes-project/
├── dataset/
│   ├── winequality-red.csv
│   └── SMSSpamCollection
├── notebooks/
│   ├── gaussian_nb_project.ipynb
│   ├── multinomial_nb_project.ipynb
│   └── bernoulli_nb_project.ipynb
├── src/
├── .gitignore
└── README.md
```

---

## Project Status

```text
Gaussian NB       ████████████████████ 100%  Completed
Multinomial NB    ████████████████████ 100%  Completed
Bernoulli NB      ██████████░░░░░░░░░░  50%  In Progress
```

---

## Future Work

Complete the **Bernoulli Naive Bayes** implementation following the same methodology (EDA → preprocessing → binary features → from-scratch model → prediction → evaluation → Scikit-learn comparison → complexity/hyperparameters → applications). Once done, all three variants will be complete, enabling a final comparison of how the choice of Naive Bayes algorithm depends on the data type and feature representation.

---

## Conclusion

This project implements and explains Naive Bayes variants from scratch using Python and NumPy. **Gaussian NB** was completed on the Wine Quality Red dataset; **Multinomial NB** achieved **97.97% accuracy** on SMS Spam Collection, matching Scikit-learn exactly; **Bernoulli NB** is in progress and will complete the set. Overall, the project covers the mathematical foundations, assumptions, implementation, feature representations, evaluation, complexity, hyperparameters, and practical applications of each variant.