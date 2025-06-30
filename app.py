from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open("Personalityrff.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form["Time_spent_Alone"]),
            int(request.form["Stage_fear"]),
            int(request.form["Social_event_attendance"]),
            int(request.form["Going_outside"]),
            int(request.form["Drained_after_socializing"]),
            int(request.form["Friends_circle_size"]),
            int(request.form["Post_frequency"]),
        ]

        prediction = model.predict([features])[0]
        personality = "Introvert 😌" if prediction == 1 else "Extrovert 😄"
        return render_template("index.html", result=f"Predicted Personality: {personality}")

    except Exception as e:
        return render_template("index.html", result=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
