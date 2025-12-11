import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


df = pd.read_csv('CSV_combined/cleaned_combined_posts.csv')
print(f"Loaded {len(df)} records from cleaned_combined_posts.csv")
print(
    f"Viral posts: {df['is_viral'].sum()}, Non-viral posts: {(df['is_viral'] == 0).sum()}")


label_encoder = LabelEncoder()
df['day_of_week_encoded'] = label_encoder.fit_transform(df['day-of-week'])


df['punctuation_count'] = df['title'].astype(str).apply(lambda x: sum(1 for c in x if c in '!?.,;:'))

tfidf = TfidfVectorizer(max_features = 2000, stop_words='english')
title_tfidf = tfidf.fit_transform(df['title'].astype(str))
title_tfidf_df = pd.DataFrame(title_tfidf.toarray(), columns=[f'title_tfidf_{i}' for i in range(title_tfidf.shape[1])])

features_df = pd.concat([
    title_tfidf_df.reset_index(drop=True),
    df[['hour-of-day', 'day_of_week_encoded', 'title_length']].reset_index(drop=True)
], axis=1)


X = features_df
y = df['is_viral']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)

joblib.dump(model, 'logistic_regression_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("\nModel trained successfully!")

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,target_names=['Non-Viral', 'Viral']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
