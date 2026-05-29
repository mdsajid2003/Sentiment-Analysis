# 🧠 Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)
![BERT](https://img.shields.io/badge/BERT-HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Twilio](https://img.shields.io/badge/Twilio-WhatsApp-F22F46?style=for-the-badge&logo=twilio&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)

> A real-time sentiment analysis web app that classifies text as **Positive**, **Negative**, or **Neutral** using a fine-tuned BERT model — with WhatsApp alerts via Twilio for negative sentiment.

---

## ✨ Features

- 🔍 **BERT-based NLP** — deep learning model for accurate sentiment classification
- 🌐 **Flask Web App** — clean UI to enter text and view results instantly
- 📲 **WhatsApp Notifications** — automatically alerts via Twilio when negative sentiment is detected
- 📊 **Twitter Dataset** — trained on 74,000+ labelled tweets
- ⚡ **Real-time Prediction** — instant results on any text input

---

## 🖥️ Demo

> Enter any text → get instant sentiment prediction → negative results trigger a WhatsApp alert

![Demo](https://via.placeholder.com/800x400?text=Add+a+screenshot+here)

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| Web Framework | Flask |
| NLP Model | BERT (HuggingFace Transformers) |
| Notifications | Twilio WhatsApp API |
| Data Processing | Pandas, NumPy |
| ML Library | Scikit-learn |
| Frontend | HTML, CSS |

---

## 📁 Project Structure

```
Sentiment-Analysis/
├── app.py                    # Flask web application
├── bert_model.py             # BERT model loading & prediction
├── train.py                  # Model training script
├── preprocess.py             # Text preprocessing pipeline
├── data_loader.py            # Dataset utilities
├── twitter_training.csv      # Training dataset (74k+ tweets)
├── twitter_validation.csv    # Validation dataset
├── index.html                # Home page UI
├── result.html               # Results page UI
├── styles.css                # Styling
└── requirements.txt
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/mdsajid2003/Sentiment-Analysis.git
cd Sentiment-Analysis

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Open your browser at `http://localhost:5000`

---

## 📊 Dataset

Twitter Entity Sentiment dataset with **74,000+ labelled tweets** across Positive, Negative, and Neutral categories.

Source: [Kaggle — Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)

---

## 🔮 Future Improvements

- [ ] Deploy on Render / HuggingFace Spaces
- [ ] Add multi-language support
- [ ] Real-time sentiment dashboard with charts
- [ ] Fine-tune on domain-specific datasets

---

## 👨‍💻 Author

**Md Sajid**
[![GitHub](https://img.shields.io/badge/GitHub-mdsajid2003-181717?style=flat&logo=github)](https://github.com/mdsajid2003)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
