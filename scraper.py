import requests
import pandas as pd
import time
import random

def scrape_askreddit(num_pages=20):
    url = "https://www.reddit.com/r/AskReddit/hot.json"
    
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 OPR/105.0.0.0'
    }
   
    after_tokens = None
    dataset = []
    
    for i in range(num_pages):
        params = {'limit': 100}
        if after_tokens:
            params['after'] = after_tokens
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"Failed to retrieve data: {response.status_code}")
            break
        data = response.json()
        posts = data['data']['children']
        for post in posts:
            post_data = post['data']
            dataset.append({
                'title': post_data['title'],
                'score': post_data['score'],
                'num_comments': post_data['num_comments'],
                'created_utc': post_data['created_utc'],
                'upvote_ratio': post_data['upvote_ratio']
            })
        after_tokens = data['data']['after']
        
        if not after_tokens:
            break
        time.sleep(random.uniform(1, 3))  # Random delay between requests
        
    df = pd.DataFrame(dataset)
    df.to_csv('askreddit_posts.csv', index=False)
    print(f"Scraped {len(df)} posts from r/AskReddit.")
    return df

if __name__ == "__main__":
    scrape_askreddit(num_pages=10)
    
