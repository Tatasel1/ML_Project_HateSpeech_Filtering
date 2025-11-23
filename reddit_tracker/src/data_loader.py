import praw
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

def get_reddit_instance():
    return praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT")
    )

def fetch_data(subreddits, limit=100):
    reddit = get_reddit_instance()
    posts = []
    for sub in subreddits:
        subreddit = reddit.subreddit(sub)
        for post in subreddit.hot(limit=limit):
            posts.append({
                "title": post.title,
                "selftext": post.selftext,
                "score": post.score,
                "created_utc": post.created_utc,
                "subreddit": sub,
                "num_comments": post.num_comments
            })
    return pd.DataFrame(posts)

def load_from_csv(filepath):
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        return pd.DataFrame()
