from flask import Flask, render_template, request 
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('food_order_model.pkl')

@app.route('/') 
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    gender = 1 if request.form['gender'] == 'Male' else 0
    family = int(request.form['family'])
    qualification = int(request.form['qualification'])
    occupation = int(request.form['occupation'])

    input_data = np.array([[age, gender, family, qualification, occupation]])
    prediction = model.predict(input_data)[0]

    result = 'Will Order Online' if prediction == 1 else 'Will Not Order Online'
    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)