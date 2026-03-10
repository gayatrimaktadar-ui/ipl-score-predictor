from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Home page
@app.route('/')
def home():
    return render_template("index.html")

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from form
        runs = int(request.form['runs'])
        wickets = int(request.form['wickets'])
        overs = float(request.form['overs'])
        runs_last_5 = int(request.form['runs_last_5'])
        wickets_last_5 = int(request.form['wickets_last_5'])

        # Convert input into numpy array
        data = np.array([[runs, wickets, overs, runs_last_5, wickets_last_5]])

        # Make prediction
        prediction = model.predict(data)

        output = int(prediction[0])

        return render_template(
            "index.html",
            prediction_text=f"Predicted Score: {output}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )

# Run the app
if __name__ == "__main__":
    app.run(debug=True)