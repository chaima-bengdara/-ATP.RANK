# -*- coding: utf-8 -*-
"""
Created on Mon May  1 19:39:01 2023

@author: chaim
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May  1 02:49:59 2023

@author: chaim
"""

import pickle
from flask import Flask, render_template, request


app = Flask(__name__, template_folder='D:/based')



@app.route('/')
def home():
     return render_template('file2.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.form.to_dict()

    # Map the keys to the names used in the trained model
    mapped_data = {
        "tourney_id": int(data["tourney_id"]),
        "surface": int(data["surface"]),
        "draw_size": int(data["draw_size"]),
        "match_num": int(data["match_num"]),
        "second_id": int(data["second_id"]),
        "second_hand": int(data["second_hand"]),
        "second_ht": int(data["second_ht"]),
        "second_ioc": int(data["second_ioc"]),
        "second_age": int(data["second_age"]),
        "first_id": int(data["first_id"]),
       "first_hand": data["first_hand"],
       "first_ht": int(data["first_ht"]),
       "first_ioc":int( data["first_ioc"]),
       "first_age": int(data["first_age"]),
       "second_ace": int(data["second_ace"]),
       "second_1stWon": int(data["second_1stWon"]),
       "second_2ndWon": int(data["second_2ndWon"]),
       "second_bpFaced": int(data["second_bpFaced"]),
       "first_ace": int(data["first_ace"]),
      "first_1stWon": int(data["first_1stWon"]),
      "first_2ndWon": int(data["first_2ndWon"]),
       "first_bpFaced": int(data["first_bpFaced"]),
      "second_rank": int(data["second_rank"]),
       "first_rank": int(data["first_rank"]),
      "tourney_year": int(data["tourney_year"]),
       "tourney_month": int(data["tourney_month"])

     
    }
      # Load the selected model from disk
    model = None
    if 'svm' in data['model']:
        with open('svm_model.pkl', 'rb') as f:
            model = pickle.load(f)
    elif 'dt' in data['model']:
        with open('dt_model.pkl', 'rb') as f:
            model = pickle.load(f)
    elif 'rf' in data['model']:
        with open('rf_model.pkl', 'rb') as f:
            model = pickle.load(f)
    elif 'gb' in data['model']:
        with open('gb_model.pkl', 'rb') as f:
            model = pickle.load(f)

       # Make a prediction using the selected model
    if model is not None:
        prediction = model.predict([list(mapped_data.values())])

        # Convert the prediction to a list and return it as a HTML response
        return render_template("prediction.html", prediction=int(prediction[0]))
    else:
        return "Please select a model"
import os
import pickle
from flask import Flask, jsonify, request, render_template
if __name__ == '__main__':
    # Get the absolute path to the directory containing this script
    basedir = os.path.abspath(os.path.dirname(__file__))

    # Start the Flask app on port 5000
    app.run(port=5001, debug=True)
