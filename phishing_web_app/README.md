# 📧 Transformer-Based Email Classification using BERT

A deep learning project for detecting phishing emails using BERT, a state-of-the-art Transformer-based NLP model. This repository includes preprocessing steps, BERT fine-tuning code, model saving utilities, and a fully functional Flask web application for real-time email classification.

## 🚀 Features

✔ Fine-tuned BERT model for phishing detection
✔ WordPiece tokenization using HuggingFace
✔ Clean dataset preprocessing pipeline
✔ Stratified train-test split
✔ Evaluation with accuracy, F1-score, and loss metrics
✔ Final model saved in final_model/
✔ Flask web app for real-time email classification
✔ Light/Dark mode UI
✔ Easy deployment and extensible architecture

## 📁 Project Structure
phishing_web_app/
│
├── app.py                  # Flask backend for prediction
├── templates/
│   └── index.html          # Frontend UI
├── static/
│   ├── style.css           # Styling (light + dark mode)
│   └── script.js           # UI interactions
│
├── final_model/            # Fine-tuned BERT model files
│   ├── config.json
│   ├── model.safetensors
│   ├── vocab.txt
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── label_classes.npy
│
├── phishing_email.csv      # Dataset (if included)
└── README.md

## 🧠 NLP Algorithms Used
1. WordPiece Tokenization

Converts email text into subword units for BERT to understand rare or complex words.

2. Transformer Encoder (BERT)

Captures deep bidirectional context to detect phishing content based on semantics, not keywords.

## 🔥 Novelty

Uses BERT for context-aware phishing detection beyond keyword-based methods

Detects sophisticated phishing emails by understanding semantic relationships in text

## 🏗️ Installation
1. Clone the Repository
git clone https://github.com/<username>/<repository-name>.git
cd phishing_web_app

2. Install Dependencies
pip install -r requirements.txt


(If you need, I can generate the requirements.txt also.)

## ▶️ Running the Web App
python app.py


Then open in your browser:

http://127.0.0.1:5000/

## 📊 Model Training Summary

Model: BERT-base-uncased

Epochs: 5

Batch Size: 16

Max Length: 128

Optimizer: AdamW

Evaluation: Accuracy & F1-score

## 📘 Results

✔ High accuracy in detecting phishing email content
✔ Strong generalization on unseen emails
✔ Clear improvement over traditional ML models

(You can add your screenshots here.)

## 🛠️ Future Enhancements

Multi-class classification

URL and attachment analysis

Metadata detection (SPF/DKIM)

Browser extension version

Cloud-based email monitoring API

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to modify.

## 📜 License

This project is licensed under the MIT License.
