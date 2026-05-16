import numpy as np
import pandas as pd
import glob  # For getting all csvs with one path
from textblob import TextBlob

# Function so analyze text sentiment for titles and text bodies
# Returns polarity and subjectivity scores as a pandas Series

def analyze_text(text):
    if pd.isna(text) or text.strip() == '':
        return pd.Series([0.0, 0.0])  # Neutral sentiment for empty or NaN text
    blob = TextBlob(text)
    return pd.Series([blob.sentiment.polarity, blob.sentiment.subjectivity])

def concat_csv_files():
    
    files = glob.glob('CSV_combined/posts_*.csv')  # Adjust the path as needed

    df_list = []
    for file in files:
        temp_df = pd.read_csv(file)
        df_list.append(temp_df)
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.to_csv('CSV_combined/combined_posts.csv', index=False)

    print(
        f"Combined {len(files)} files into 'combined_posts.csv' with {len(combined_df)} total records.")

    combined_df.drop_duplicates(subset='id', inplace=True)
    combined_df.dropna(subset=['title'], inplace=True)

    # date conversion

    combined_df['created_utc'] = pd.to_datetime(combined_df['created_utc'], unit='s')

    now = pd.Timestamp.now(tz='UTC')

    if combined_df['created_utc'].dt.tz is None:
        combined_df['created_utc'] = combined_df['created_utc'].dt.tz_localize('UTC')

    combined_df['age_hours'] = (now - combined_df['created_utc']).dt.total_seconds() / 3600.0
    original_count = len(combined_df)
    combined_df = combined_df[combined_df['age_hours'] > 6]

    combined_df['hour-of-day'] = combined_df['created_utc'].dt.hour
    combined_df['day-of-week'] = combined_df['created_utc'].dt.day_name()

    combined_df['title_length'] = combined_df['title'].astype(str).apply(len)
    
    print("Analyzing text senriment for titles and text bodies")
    
    # Replace NaN in selftext with empty string
    combined_df['selftext'] = combined_df.get('selftext', '').fillna('')
    
    combined_df[['title_polarity', 'title_subjectivity']] = combined_df['title'].apply(analyze_text)
    combined_df[['body_polarity', 'body_subjectivity']] = combined_df['selftext'].apply(analyze_text)
    combined_df['body_length'] = combined_df['selftext'].astype(str).apply(len)
    
    # Count uppercase words in title
    combined_df['uppercase_word_count'] = combined_df['title'].astype(str).apply(
        lambda x: sum(1 for word in x.split() if word.isupper() and len(word) > 1)
    )

    # Score add field is_viral
    # if score > 50 then is_viral = 1 else 0
    # IF ONLY /HOT CHANGE SCORE TO PERCENTILE

    viral_threshold = combined_df['score'].quantile(0.75)
    combined_df['is_viral'] = combined_df['score'].apply(
        lambda x: 1 if x >= viral_threshold else 0)

    columns_to_drop = ['domain', 'is_self', 'created_utc', 'upvote_ratio', 'num_comments', 'score','id', 'age_hours']
    existing_columns = [col for col in columns_to_drop if col in combined_df.columns]
    combined_df.drop(columns=existing_columns, inplace=True)
    
    combined_df.to_csv('CSV_combined/cleaned_combined_posts.csv', index=False)
    print(
        f"Cleaned data saved to 'CSV_combined/cleaned_combined_posts.csv' with {len(combined_df)} records after cleaning.")


if __name__ == "__main__":
    concat_csv_files()
