# Reddit Trend Tracker 🚀

A Python-based machine learning project to track and visualize trends on Reddit.

## Features

- **Data Collection**: Fetches hot posts from specified subreddits using PRAW.
- **CSV Mode**: Load data from a local CSV file for offline testing.
- **Sentiment Analysis**: Analyzes the sentiment of post titles using NLTK VADER.
- **Topic Modeling**: Extracts trending topics using NMF (Non-negative Matrix Factorization).
- **Interactive Dashboard**: Desktop GUI built with CustomTkinter and Matplotlib.

## Setup

1.  **Clone the repository** (if applicable) or navigate to the project folder.
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set up Environment Variables**:
    - Copy `.env.example` to `.env`:
      - Windows (PowerShell): `Copy-Item .env.example .env`
      - Linux/Mac: `cp .env.example .env`
    - Open `.env` and fill in your Reddit API credentials:
      - `REDDIT_CLIENT_ID`
      - `REDDIT_CLIENT_SECRET`
      - `REDDIT_USER_AGENT`
    - _Note: You can get these by creating an app at [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)._

## Running the App

Run the desktop application:

```bash
python src/app.py
```

## Project Structure

- `src/app.py`: Main desktop application (CustomTkinter).
- `src/data_loader.py`: Handles Reddit API interactions.
- `src/preprocessor.py`: Cleans and prepares text data.
- `src/analyzer.py`: Performs sentiment analysis and topic modeling.
