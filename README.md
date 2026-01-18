# Reddit Virality Prediction

A machine learning project that predicts whether Reddit posts will go viral based on posting-time features using logistic regression.

## Overview

This project scrapes Reddit posts from multiple subreddits, preprocesses the data, and trains a classification model to predict post virality (top 25% by score) using features available at the time of posting.

## Features

The model uses the following features:

- **Title Text**: TF-IDF vectorization (100 features)
- **Temporal**: Hour-of-day, day-of-week
- **Content**: Title length, is_self_post (text vs. link)

**Target**: Posts are classified as "viral" if their score is in the top 25% (75th percentile).

## Project Structure

```
├── data_collection.ipynb          # Reddit scraping notebook
├── data_preprocessing.ipynb       # Data cleaning and feature engineering
├── reddit_virality_analysis.ipynb # Model training and evaluation
├── scraper.py                     # Reddit API scraper
├── concatCsv.py                   # Data preprocessing script
├── model.py                       # Logistic regression model
├── CSV_combined/                  # Scraped data directory
└── requirements.txt               # Python dependencies
```

## Setup

1. **Create virtual environment**:

```powershell
py -m venv venv
venv\Scripts\Activate.ps1
```

2. **Install dependencies**:

```powershell
pip install -r requirements.txt
```

## Usage

### Option 1: Run Web App (Recommended)
1. Navigate to the app folder
2. Type the following command into the terminal:
   ```powershell
   python app.py
   ```
3. Navigate to one of the web adresses shown into the terminal

**In order to observe the accuracy of the model either navigate to the web application's dedicated page or use the following options.**
   
### Option 2: Run Jupyter Notebooks

1. **Collect Data**: Run `data_collection.ipynb`
2. **Preprocess**: Run `data_preprocessing.ipynb`
3. **Train & Evaluate**: Run `reddit_virality_analysis.ipynb`

### Option 3: Run Python Scripts

```powershell
# 1. Scrape Reddit data
python scraper.py

# 2. Preprocess data
python concatCsv.py

# 3. Train model
python model.py
```

## Data Collection

The scraper targets 15 subreddits across categories:

- Humor: funny, pics, todayilearned, wholesomememes
- Discussion: AskReddit, NoStupidQuestions, AmItheAsshole, tifu
- News: worldnews, technology, science
- Culture: gaming, movies, wallstreetbets, Music

## Model Performance

The logistic regression model includes:

- StandardScaler for feature normalization
- TF-IDF vectorization for text features
- Label encoding for categorical features
- Train/test split (80/20)

Outputs:

- Model accuracy and classification metrics
- Confusion matrix
- ROC curve
- Feature importance analysis

## Output Files

- `CSV_combined/cleaned_combined_posts.csv` - Preprocessed data
- `model.pkl` - Trained logistic regression model
- `tfidf_vectorizer.pkl` - Fitted TF-IDF vectorizer
- `label_encoder.pkl` - Day-of-week encoder
- `scaler.pkl` - Feature scaler

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- requests
- joblib

## Notes

- Viral threshold is dynamic (75th percentile of scores)
- Features avoid data leakage (no num_comments)
- Rate limiting implemented for Reddit API compliance

