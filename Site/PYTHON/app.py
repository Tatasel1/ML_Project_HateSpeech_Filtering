import os
import pandas as pd
from flask import Flask, request, render_template
from dotenv import load_dotenv
import joblib
from textblob import TextBlob

load_dotenv()

app = Flask(__name__, template_folder='../HTML', static_folder='../CSS', static_url_path='/CSS')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    model_path = os.path.join(BASE_DIR, 'logistic_regression_model.pkl')
    tfidf_path = os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl')
    scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')

    model = joblib.load(model_path)
    tfidf = joblib.load(tfidf_path)
    scaler = joblib.load(scaler_path)
    print("Model încărcat cu succes!")

except FileNotFoundError as e:
    model = None
    print(f"EROARE: Nu am gasit fisierele. Python cauta aici: {BASE_DIR}")
    print(f"Detalii eroare: {e}")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analysis')
def analysis():
    return render_template('plots.html')


@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return "Modelul nu este disponibil. Verifică logurile serverului pentru detalii.", 500

    title = request.form['title']
    body = request.form.get('body', '')
    hour_of_day = int(request.form['hour_of_day'])
    day_of_week = request.form['day_of_week']

    title_length = len(title)
    body_length = len(body)
    
    punctuation_count = sum(1 for c in title if c in '!?.,;:')
    uppercase_word_count = sum(1 for word in title.split() if word.isupper() and len(word) > 1)
    
    def get_sentiment(text):
        if not text or text.strip() == '':
            return 0.0, 0.0
        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    
    title_polarity, title_subjectivity = get_sentiment(title)
    body_polarity, body_subjectivity = get_sentiment(body)
    
    title_vect = tfidf.transform([title])
    tfidf_columns = tfidf.get_feature_names_out()
    tfidf_df = pd.DataFrame(title_vect.toarray(), columns=tfidf_columns)
    
    numerical_columns = [
        'hour-of-day', 'title_length', 'punctuation_count',
        'title_polarity', 'title_subjectivity', 'uppercase_word_count',
        'body_polarity', 'body_subjectivity', 'body_length'
    ]
    
    num_df = pd.DataFrame([[
        hour_of_day, title_length, punctuation_count,
        title_polarity, title_subjectivity, uppercase_word_count,
        body_polarity, body_subjectivity, body_length
    ]], columns=numerical_columns)
    
    scaled_num_df = pd.DataFrame(scaler.transform(num_df), columns=numerical_columns)
    
    expected_features = model.feature_names_in_
    
    input_data = {feat: 0.0 for feat in expected_features}
    
    for col, val in zip(tfidf_columns, title_vect.toarray()[0]):
        if col in input_data:
            input_data[col] = val
            
    for col, val in zip(numerical_columns, scaled_num_df.iloc[0]):
        if col in input_data:
            input_data[col] = val
            
    day_col_name = f"day-of-week_{day_of_week}"
    if day_col_name in input_data:
        input_data[day_col_name] = 1.0
    
    final_input_df = pd.DataFrame([input_data])[expected_features]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(final_input_df)[0]
        confidence_val = probs[1] * 100
        confidence = f"{confidence_val:.1f}%"
    else:
        confidence_val = 0
        confidence = "N/A"

    if confidence_val >= 60:
        result_message = f"VIRAL! (Chance: {confidence})"
        alert_type = "success"

    elif confidence_val >= 40:
        result_message = f"Potentially Viral (Chance: {confidence})"
        alert_type = "warning"

    else:
        result_message = f"Not Viral (Chance: {confidence})"
        alert_type = "secondary"

    return render_template('index.html', 
                           prediction=result_message, 
                           alert_type=alert_type, 
                           original_title=title, 
                           original_body=body, 
                           original_hour=hour_of_day, 
                           original_day=day_of_week)


if __name__ == '__main__':
    app.run(debug=True)

    # SI RANDOM FOREST
