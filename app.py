from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)
print("THIS IS MY APP.PY")

model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/info")
def info():
    return render_template("info.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        age = int(request.form["age"])
        gender = request.form["gender"]
        occupation = request.form["occupation"]
        sleep_duration = float(request.form["sleep_duration"])
        quality_of_sleep = int(request.form["quality_of_sleep"])
        physical_activity = float(request.form["physical_activity"])
        stress_level = int(request.form["stress_level"])
        bmi = float(request.form["bmi"])
        heart_rate = int(request.form["heart_rate"])
        daily_steps = int(request.form["daily_steps"])
        sleep_efficiency = float(request.form["sleep_efficiency"])
        bmi_category = request.form["bmi_category"]

        user_data = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "Occupation": occupation,
            "Sleep_duration": sleep_duration,
            "Quality_of_sleep": quality_of_sleep,
            "Physical_activity": physical_activity,
            "Stress_Level": stress_level,
            "BMI": bmi,
            "Heart_rate": heart_rate,
            "Daily_steps": daily_steps,
            "Sleep_Efficiency": sleep_efficiency,
            "BMI_category": bmi_category
        }])

        categorical_columns = [
            "Gender",
            "Occupation",
            "BMI_category"
        ]

        for col in categorical_columns:
            user_data[col] = encoders[col].transform(user_data[col])

        prediction = model.predict(user_data)

        result = target_encoder.inverse_transform(prediction)[0]

        return render_template(
            "result.html",
            prediction=result
        )

    return render_template("predict.html")

print(app.url_map)

if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host="127.0.0.1", port=8000, debug=True)

