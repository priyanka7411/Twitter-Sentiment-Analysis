# 🐦 Twitter Sentiment Analysis Dashboard

A comprehensive web application for analyzing sentiment in tweets using Natural Language Processing (NLP) techniques.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Project Overview

This project analyzes sentiments from tweets using two popular NLP methods:
- **TextBlob**: Pattern-based sentiment analysis
- **VADER**: Specially designed for social media sentiment analysis

## ✨ Features

- **Single Tweet Analysis**: Analyze individual tweets in real-time
- **Bulk CSV Upload**: Process multiple tweets at once
- **Dual Analysis Methods**: Compare TextBlob and VADER results
- **Visual Analytics**: 
  - Interactive gauge charts
  - Sentiment distribution pie charts
  - Word clouds
  - Bar charts
- **Text Preprocessing**: Automatic cleaning of URLs, mentions, hashtags
- **Export Results**: Download analyzed data as CSV

## 🛠️ Technologies Used

- **Python 3.11+**
- **Streamlit**: Web application framework
- **TextBlob**: NLP library for sentiment analysis
- **VADER Sentiment**: Social media sentiment analysis
- **Plotly**: Interactive visualizations
- **Pandas & NumPy**: Data manipulation
- **WordCloud**: Text visualization
- **NLTK**: Natural Language Toolkit

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK data
```bash
python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"
```

## 🚀 Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📊 How to Use

### Single Tweet Analysis
1. Go to the "📝 Single Tweet" tab
2. Enter your tweet text
3. Select analysis method (TextBlob, VADER, or Both)
4. Click "🔍 Analyze Sentiment"

### Bulk Analysis
1. Go to the "📁 Bulk Analysis" tab
2. Upload a CSV file with a 'text' or 'tweet' column
3. Click "🚀 Start Analysis"
4. View results and download analyzed data

### Sample Dataset
1. Go to the "📊 Sample Dataset" tab
2. Click "🎲 Generate Sample Data"
3. Explore pre-loaded examples

## 📂 Project Structure
```
Project1_Twitter_Sentiment/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── .gitignore             # Git ignore file
│
├── data/                  # Data files
│   └── sample_tweets.csv
│
├── notebooks/             # Jupyter notebooks (for analysis)
├── models/                # Saved models (if any)
└── venv/                  # Virtual environment
```

## 🎯 Key Learnings

- Natural Language Processing fundamentals
- Sentiment analysis techniques
- Text preprocessing and cleaning
- Building interactive web applications with Streamlit
- Data visualization with Plotly
- Working with multiple NLP libraries

## 🔮 Future Enhancements

- [ ] Real-time Twitter API integration
- [ ] Multi-language support
- [ ] Advanced visualizations (trend analysis)
- [ ] Deep learning models (BERT, RoBERTa)
- [ ] User authentication and history
- [ ] Emoji sentiment analysis

## 👩‍💻 Author

**Priyanka Malavade**
- BCA Graduate 2024
- Data Science Enthusiast
- Portfolio Project

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- TextBlob documentation
- VADER Sentiment Analysis
- Streamlit community
- GUVI Data Science Course

---

⭐ If you found this project helpful, please give it a star!