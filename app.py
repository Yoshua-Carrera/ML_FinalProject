# This is the app
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('models/svm.pkl', 'rb'))

@app.route('/')
def home():
        return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    :return:
    '''

@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    :return:
    '''

if __name__ == "__main__":
    app.run(debug=True)

