import streamlit as st
import pandas as pd
import plotly.express as px
from data_loader import fetch_data
from preprocessor import clean_text
from analyzer import analyze_sentiment, extract_topics

def main():
    st.set_page_config(page_title="Reddit Trend Tracker", layout="wide")
    st.title("Reddit Trend Tracker 🚀")

    # Sidebar for configuration
    st.sidebar.header("Configuration")
    subreddits_input = st.sidebar.text_input("Subreddits (comma separated)", "technology,python,datascience")
    subreddits = [s.strip() for s in subreddits_input.split(",")]
    limit = st.sidebar.slider("Max Posts per Subreddit", 10, 500, 100)
    
    if st.sidebar.button("Fetch & Analyze"):
        with st.spinner("Fetching data from Reddit..."):
            try:
                df = fetch_data(subreddits, limit=limit)
                if df.empty:
                    st.error("No data found. Check your API credentials or subreddit names.")
                    return
                
                st.success(f"Fetched {len(df)} posts!")
                
                # Preprocessing
                with st.spinner("Analyzing..."):
                    df['cleaned_text'] = df['title'].apply(clean_text)
                    df = analyze_sentiment(df)
                    
                    # Display Data
                    st.subheader("Recent Posts")
                    st.dataframe(df[['subreddit', 'title', 'score', 'sentiment']].head())
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Sentiment Distribution")
                        fig_hist = px.histogram(df, x="sentiment", nbins=20, title="Sentiment Score Distribution", color_discrete_sequence=['#636EFA'])
                        st.plotly_chart(fig_hist, use_container_width=True)
                        
                    with col2:
                        st.subheader("Average Sentiment by Subreddit")
                        avg_sentiment = df.groupby("subreddit")['sentiment'].mean().reset_index()
                        fig_bar = px.bar(avg_sentiment, x="subreddit", y="sentiment", title="Avg Sentiment", color="sentiment", color_continuous_scale="RdBu")
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # Topic Modeling
                    st.subheader("Trending Topics")
                    topics = extract_topics(df)
                    for topic, words in topics.items():
                        st.write(f"**{topic}:** {', '.join(words)}")
                        
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
