import requests
import pandas as pd
import time
import random

# General / Humor
# 'funny', 'pics', 'todayilearned', 'wholesomememes',
    
# Text / Stories
#  'AskReddit', 'NoStupidQuestions', 'AmItheAsshole', 'tifu',
    
# News / Info
#  'worldnews', 'technology', 'science',
    
# Hype / Culture
#  'gaming', 'movies', 'wallstreetbets', 'Music'


def scrape_askreddit(subreddits, num_pages = 20 , list_type = 'hot'):
    
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 OPR/105.0.0.0'
    }
    for sub in subreddits:
        url = f"https://www.reddit.com/r/{sub}/{list_type}.json"
        after_tokens = None
        dataset = []
    
        for i in range(num_pages):
            
            params = {'limit': 100}
            
            if after_tokens:
                params['after'] = after_tokens
            
            attempt = 0
            while attempt < 3:
                    response = requests.get(url, headers=headers, params=params)
                    
                    if response.status_code == 429:
                        time.sleep(60)
                        attempt += 1
                        continue
                    elif response.status_code != 200:
                        print(f"Failed to retrieve data: {response.status_code}")
                        break 
                    else:
                        break
            
            if response.status_code != 200:
                print(f"Failed to retrieve data: {response.status_code}")
                break
            
            data = response.json()
            
            posts = data['data']['children']
            
            for post in posts:
                post_data = post['data']
                dataset.append({
                    
                    'id': post_data['id'],
                    'subreddit': post_data['subreddit'],
                    'title': post_data['title'],
                    'score': post_data['score'],
                    'num_comments': post_data['num_comments'],
                    'created_utc': post_data['created_utc'],
                    'upvote_ratio': post_data['upvote_ratio'],
                    'is_self': post_data['is_self'],
                    'domain': post_data['domain']
                    
                })
                
            after_tokens = data['data']['after']
        
            if not after_tokens:
                break
            time.sleep(random.uniform(1, 3))  # Random delay between requests
        if dataset:
            df = pd.DataFrame(dataset)
            df.to_csv(f'posts_{sub}_{list_type}.csv', index=False)
            print(f"Scraped {len(df)} posts from r/{sub}.")
            
        time.sleep(random.uniform(2, 5))  # Random delay between subreddits
    return df

if __name__ == "__main__":
    
    target_subreddits = [
    'funny', 'pics', 'todayilearned', 'wholesomememes',
    'AskReddit', 'NoStupidQuestions', 'AmItheAsshole', 'tifu','worldnews', 
    'technology', 'science',
    'gaming', 'movies', 'wallstreetbets', 'Music'
]
    scrape_askreddit(target_subreddits, num_pages=20, list_type='hot')
    
    scrape_askreddit(target_subreddits, num_pages=20, list_type='new')