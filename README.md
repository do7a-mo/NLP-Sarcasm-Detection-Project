# NLP-Sarcasm-Detection-Project
NLP project for sarcasm detection in news headlines using TF-IDF and multiple machine learning classifiers including RidgeClassifier, LinearSVC, SGDClassifier, Logistic Regression, and Multinomial Naive Bayes.
## Overview
This project applies Natural Language Processing (NLP) and Machine Learning techniques to classify news headlines as sarcastic or non-sarcastic.

The workflow includes:
- Data preprocessing
- Text cleaning
- Feature engineering
- TF-IDF vectorization
- Training multiple ML models
- Performance evaluation and comparison

---
## Dataset
The dataset used in this project is:
- Sarcasm News Headlines Dataset
- JSON format

Main columns:
- `headline` → News headline text
- `is_sarcastic` → Target label (0 = Non-Sarcastic, 1 = Sarcastic)

---

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- NLP Techniques
- TF-IDF Vectorization
- Machine Learning Models

---

## Data Preprocessing
The preprocessing pipeline includes:
- Removing duplicates
- Text cleaning
- Lowercasing
- Removing unnecessary characters
- Feature engineering

Additional engineered features:
- Text length
- Word count
- Exclamation/question marks count
- Uppercase ratio

---

## Models Used
The following machine learning models were trained and evaluated:

1. RidgeClassifier
2. LinearSVC
3. SGDClassifier
4. Logistic Regression
5. Multinomial Naive Bayes

---

## Evaluation Metrics
The models were evaluated using:
- Accuracy
- F1-Score

---

## Project Workflow
1. Load dataset
2. Clean and preprocess text
3. Apply TF-IDF vectorization
4. Split dataset into training/testing sets
5. Train multiple classifiers
6. Evaluate model performance
7. Compare results

---
## Example Pipeline
```python
Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression())
])
