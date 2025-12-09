import numpy as np 
import pandas as pd
import glob # For getting all csvs with one path

def concat_csv_files():
    files = glob.glob('CSVs/posts_*.csv')  # Adjust the path as needed

    df_list = []
    for file in files:
        temp_df = pd.read_csv(file)
        df_list.append(temp_df)
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.to_csv('combined_posts.csv', index=False)
    
    print(f"Combined {len(files)} files into 'combined_posts.csv' with {len(combined_df)} total records.")

    combined_df.drop_duplicates(subset='id', inplace=True)
    combined_df.dropna(inplace=True)
    #date conversion
    combined_df['created_utc'] = pd.to_datetime(combined_df['created_utc'], unit='s')
    combined_df['hour-of-day'] = combined_df['created_utc'].dt.hour
    combined_df['day-of-week'] = combined_df['created_utc'].dt.day_name()
    combined_df['month'] = combined_df['created_utc'].dt.month_name()
    # Score add field is_viral
    #if score > 50 then is_viral = 1 else 0

    score = 50
    combined_df['is_viral'] = combined_df['score'].apply(lambda x: True if x > score else False)

    combined_df.to_csv('cleaned_combined_posts.csv', index=False)
    print(f"Cleaned data saved to 'cleaned_combined_posts.csv' with {len(combined_df)} records after cleaning.")
    
if __name__ == "__main__":
    concat_csv_files()