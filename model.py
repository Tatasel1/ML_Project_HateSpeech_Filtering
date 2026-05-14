import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


df = pd.read_csv('CSV_combined/cleaned_combined_posts.csv')
print(f"Loaded {len(df)} records from cleaned_combined_posts.csv")
print(
    f"Viral posts: {df['is_viral'].sum()}, Non-viral posts: {(df['is_viral'] == 0).sum()}")


# Encode day of week
df = pd.get_dummies(df, columns=['day-of-week'], drop_first=True)

df['punctuation_count'] = df['title'].astype(str).apply(lambda x: sum(1 for c in x if c in '!?.,;:'))

# tfidf = TfidfVectorizer(max_features = 2000, stop_words='english')
# title_tfidf = tfidf.fit_transform(df['title'].astype(str))
# title_tfidf_df = pd.DataFrame(title_tfidf.toarray(), columns=[f'title_tfidf_{i}' for i in range(title_tfidf.shape[1])])

# features_df = pd.concat([
#     title_tfidf_df.reset_index(drop=True),
#     df[['hour-of-day', 'day_of_week_encoded', 'title_length']].reset_index(drop=True)
# ], axis=1)

y = df['is_viral']
X_raw = df.drop(columns = ['is_viral'])

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set: {len(X_train_raw)} samples")
print(f"Test set: {len(X_test_raw)} samples")

tfidf = TfidfVectorizer(max_features=2000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train_raw['title'].astype(str))
X_test_tfidf = tfidf.transform(X_test_raw['title'].astype(str))

tfidf_columns = tfidf.get_feature_names_out()
X_train_tfidf_df = pd.DataFrame(X_train_tfidf.toarray(), columns=tfidf_columns, index=X_train_raw.index)
X_test_tfidf_df = pd.DataFrame(X_test_tfidf.toarray(), columns=tfidf_columns, index=X_test_raw.index)

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

numerical_columns = [
    'hour-of-day', 'title_length', 'punctuation_count',
    'title_polarity', 'title_subjectivity', 'uppercase_word_count',
    'body_polarity', 'body_subjectivity', 'body_length'
]

scaler = StandardScaler()
X_train_scaled_num = pd.DataFrame(scaler.fit_transform(X_train_raw[numerical_columns]), columns=numerical_columns, index=X_train_raw.index)
X_test_scaled_num = pd.DataFrame(scaler.transform(X_test_raw[numerical_columns]), columns=numerical_columns, index=X_test_raw.index)

day_columns = [col for col in X_train_raw.columns if col.startswith('day-of-week_')]

X_train = pd.concat([X_train_tfidf_df, X_train_scaled_num, X_train_raw[day_columns]], axis=1)
X_test = pd.concat([X_test_tfidf_df, X_test_scaled_num, X_test_raw[day_columns]], axis=1)

model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

joblib.dump(model, 'logistic_regression_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("\nModel trained successfully!")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,target_names=['Non-Viral', 'Viral']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
