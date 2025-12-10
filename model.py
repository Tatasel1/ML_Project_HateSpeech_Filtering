import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the cleaned combined posts data
df = pd.read_csv('CSVs/cleaned_combined_posts.csv')

print(f"Loaded {len(df)} records from cleaned_combined_posts.csv")
print(
    f"Viral posts: {df['is_viral'].sum()}, Non-viral posts: {(df['is_viral'] == 0).sum()}")

# Features: title (text), hour-of-day (numeric), day-of-week (categorical), num_comments (numeric)
# Target: is_viral

# 1. Encode day-of-week
label_encoder = LabelEncoder()
df['day_of_week_encoded'] = label_encoder.fit_transform(df['day-of-week'])

# 2. TF-IDF vectorization for title
tfidf = TfidfVectorizer(max_features=100, stop_words='english')
title_tfidf = tfidf.fit_transform(df['title'].astype(str))
title_tfidf_df = pd.DataFrame(title_tfidf.toarray(), columns=[
                              f'title_tfidf_{i}' for i in range(title_tfidf.shape[1])])

# 3. Combine features
features_df = pd.concat([
    title_tfidf_df.reset_index(drop=True),
    df[['hour-of-day', 'day_of_week_encoded',
        'num_comments']].reset_index(drop=True)
], axis=1)

# 4. Prepare X and y
X = features_df
y = df['is_viral']

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# 6. Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Train Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

print("\nModel trained successfully!")

# 8. Make predictions
y_pred = model.predict(X_test_scaled)

# 9. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Non-Viral', 'Viral']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
