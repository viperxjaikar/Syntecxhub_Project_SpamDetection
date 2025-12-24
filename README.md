# Spam Detection — Syntecxhub Internship Project

This project implements a spam detection system using classical machine learning techniques as part of the Syntecxhub Internship program.

## Objective
- Load a labeled spam/ham dataset
- Preprocess text data
- Convert text into numerical vectors using TF-IDF
- Train a classification model
- Evaluate using precision, recall, and F1-score
- Save the trained pipeline for reuse

## Dataset
- ~1000 messages
- Labels: spam, ham
- Realistic language patterns with controlled label noise

## Approach
- Text cleaning (lowercasing, noise removal)
- TF-IDF vectorization
- Logistic Regression classifier
- Pipeline-based training and persistence

## Evaluation Metrics
- Precision
- Recall
- F1-score

Results are high but non-perfect, reflecting realistic data behavior.

## How to Run
python preprocess.py
python train_evaluate.py
python predict.py


## Files
- dataset.csv — raw dataset
- cleaned_data.csv — preprocessed data
- preprocess.py — cleaning and encoding
- train_evaluate.py — training and evaluation
- predict.py — inference
- spam_model.pkl — saved model pipeline