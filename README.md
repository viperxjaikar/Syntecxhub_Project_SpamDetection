# Spam Detection (ML Pipeline)

A machine learning-based spam detection system that classifies text messages as spam or ham using TF-IDF vectorization and logistic regression, built as part of the Syntecxhub Internship.

---

## 🎯 Objective

- Load and preprocess labeled text data  
- Convert text into numerical features  
- Train a classification model  
- Evaluate using standard metrics  
- Save the trained pipeline for reuse  

---

## ⚙️ Methodology

- Text preprocessing (lowercasing, noise removal)  
- TF-IDF vectorization for feature extraction  
- Logistic Regression for classification  
- Pipeline-based training and model persistence  

---

## 📊 Dataset

- ~1000 labeled messages  
- Classes: spam, ham  
- Contains realistic text patterns with some label noise  

---

## 🏗️ Workflow

Raw Text → Cleaning → TF-IDF → Model Training → Evaluation → Model Saving → Prediction

---

## 📁 Project Structure

spam-detection/  
├── dataset.csv              # Raw dataset  
├── cleaned_data.csv         # Preprocessed dataset  
├── preprocess.py            # Data cleaning  
├── train_evaluate.py        # Training + evaluation  
├── predict.py               # Inference  
├── spam_model.pkl           # Saved pipeline  
└── README.md  

---

## 🛠️ Tech Stack

- Python  
- Pandas  
- Scikit-learn  
- TF-IDF (feature extraction)  
- Logistic Regression  

---

## 🚀 How to Run

python preprocess.py  
python train_evaluate.py  
python predict.py  

---

## 🧪 Evaluation Metrics

- Precision  
- Recall  
- F1-score  

Results are high but not perfect, reflecting real-world noisy data.

---

## ⚠️ Limitations

- Small dataset (~1000 samples) limits generalization  
- Classical ML approach (no deep learning or NLP models)  
- No real-time deployment or API integration  
- Limited feature engineering beyond TF-IDF  

---

## 💡 Future Improvements

- Use larger, real-world datasets  
- Try advanced NLP models (BERT, LSTM)  
- Deploy as REST API or web app  
- Add continuous retraining with new data  
- Improve preprocessing (stemming, lemmatization)  

---

## 📌 Why This Project Matters

This project demonstrates:
- End-to-end ML pipeline development  
- Text preprocessing and feature engineering  
- Model evaluation using proper metrics  
- Pipeline saving and reuse  

---

## 👤 Author

Jaikar Ramu  
https://github.com/viperxjaikar  

---

## ⭐ Star if useful
