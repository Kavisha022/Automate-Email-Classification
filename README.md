# 📧 Automated Email Classification and Routing System
This project automates the classification of incoming emails into predefined insurance categories (e.g., Automobile, Medical, Housing) using BERT and SVM models. It also sends automated acknowledgment responses to customers and routes categorized emails to the relevant department.

## 🔧 Features
✅ Email Fetching from inbox using IMAP

✅ Text Cleaning for subject & body

✅ Email Parsing with mailparser

✅ BERT-based Classification using Hugging Face Transformers

✅ SVM-based Backup Classifier using TF-IDF

✅ Confusion Matrix Evaluation

✅ Automatic Acknowledgment Email

✅ Departmental Email Routing

✅ Excel Logging of processed emails

✅ Avoids Resending using Log Tracking

## 🧠 Model Training (BERT)
Tokenizes using BERT

Fine-tunes bert-base-uncased on your email data

Saves model and tokenizer to local folders

You can also optionally use an SVM classifier (svm_classifier.py) for fast training and baseline comparison.

## 📥 Fetching and Logging Emails
Extracts sender, subject, and body

Saves emails to emails.xlsx

## 📬 Sending Acknowledgment Emails
python acknowledgment_sender.py

## 📤 Forward Classified Emails to Departments
python classify_and_forward.py
Loads the BERT model

Classifies emails from Excel

Sends them to department email addresses

Uses sent_log.txt to prevent duplicates

## 📊 Evaluation
After training, train_classifier.py displays a confusion matrix and prints validation accuracy.

