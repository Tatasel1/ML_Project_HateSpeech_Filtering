from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import nltk

nltk.download('vader_lexicon', quiet=True)

def analyze_sentiment(df, text_column='title'):
    sia = SentimentIntensityAnalyzer()
    df['sentiment'] = df[text_column].apply(lambda x: sia.polarity_scores(x)['compound'])
    return df

def extract_topics(df, text_column='cleaned_text', n_topics=5):
    tfidf = TfidfVectorizer(max_features=1000, min_df=2, stop_words='english')
    dtm = tfidf.fit_transform(df[text_column])
    nmf = NMF(n_components=n_topics, random_state=42)
    nmf.fit(dtm)
    
    topics = {}
    feature_names = tfidf.get_feature_names_out()
    for index, topic in enumerate(nmf.components_):
        top_words = [feature_names[i] for i in topic.argsort()[-10:]]
        topics[f"Topic {index+1}"] = top_words
    return topics
