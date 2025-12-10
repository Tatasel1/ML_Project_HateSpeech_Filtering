import pandas as pd
from flask import Flask, request, render_template
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__, template_folder='../HTML')

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host ="0.0.0.0", port = "5000", debug=True)