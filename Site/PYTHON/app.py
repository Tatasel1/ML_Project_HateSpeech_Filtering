import pandas as pd
from flask import Flask, request, render_template
from dotenv import load_dotenv
import joblib

load_dotenv()

app = Flask(__name__, template_folder='../HTML',
            static_folder='../CSS', static_url_path='/CSS')

try:
    model = joblib.load('../../Model_and_Scaler/logistic_regression_model.pkl')
    tdidf = joblib.load('../../Model_and_Scaler/tfidf_vectorizer.pkl')
    scaler = joblib.load('../../Model_and_Scaler/scaler.pkl')
    print("Model încărcat cu succes!")

except FileNotFoundError:
    model = None
    print("EROARE: Nu am găsit 'logistic_regression_model.pkl'. Asigură-te că este în același folder.")


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
    hour_of_day = int(request.form['hour_of_day'])
    day_of_week = request.form['day_of_week']

    title_length = len(title)

    day_mapping = {
        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
        'Friday': 4, 'Saturday': 5, 'Sunday': 6
    }

    day_of_week_encoded = day_mapping.get(day_of_week, 0)

    title_vect = tdidf.transform([title]).toarray()
    title_features = pd.DataFrame(
        title_vect, columns=[f'title_tfidf_{i}' for i in range(title_vect.shape[1])])

    input_df = pd.DataFrame([{
        'hour-of-day': hour_of_day,
        'day_of_week_encoded': day_of_week_encoded,
        'title_length': title_length
    }])

    input_df = pd.concat([title_features, input_df], axis=1)
    input_df_scaled = scaler.transform(input_df)

    viral_chance_percent = 0

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_df_scaled)[0]

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

    return render_template('index.html', prediction=result_message, alert_type=alert_type, original_title=title, original_hour=hour_of_day, original_day=day_of_week)


if __name__ == '__main__':
    app.run(debug=True)

    # SI RANDOM FOREST
