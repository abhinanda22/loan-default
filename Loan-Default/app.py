import pickle
import numpy as np

from flask import Flask, request, render_template

app = Flask(__name__)

with open('xgb.pkl','rb') as file:
    model = pickle.load(file)
    
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    prediction_text = None
    if request.method == 'POST':
        loan_type = int(request.form['loan_type'])
        loan_purpose = int(request.form['loan_purpose'])
        loan_amount = float(request.form['loan_amount'])
        rate_of_interest = float(request.form['rate_of_interest'])
        term = int(request.form['term'])
        income = float(request.form['income'])
        Credit_Score = int(request.form['Credit_Score'])
        age = int(request.form['age'])

        # Convert user inputs to a numpy array for prediction
        sample_data = np.array([[loan_type, loan_purpose, loan_amount, rate_of_interest, term, income, Credit_Score, age]])

        prediction = model.predict(sample_data)
        if prediction == 0:
            prediction_text = "The probabilty of loan repayment is high, i.e, Not Default"
        else:
            prediction_text = "The probabilty of loan repayment is low, i.e, Default"

    return render_template("result.html", prediction_text=prediction_text)

#if __name__ == "__main__":
#   app.run(debug=True)