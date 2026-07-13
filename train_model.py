import pandas as pd
import xgboost as xgb
import warnings
import joblib
from preprocess import preprocess

warnings.filterwarnings("ignore")

X, y, encoders, target_encoder = preprocess()

xgb_model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42
)
def predict_sleep_disorder(age, gender, occupation, sleep_duration,
                           quality_of_sleep, physical_activity, stress_level,
                           bmi,blood_pressure, heart_rate, daily_steps,
                           bmi_category):

    user_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Occupation": occupation,
        "Sleep_duration": sleep_duration,
        "Quality_of_sleep": quality_of_sleep,
        "Physical_activity": physical_activity,
        "Stress_Level": stress_level,
        "Blood_Pressure":blood_pressure,
        "Heart_rate": heart_rate,
        "Daily_steps": daily_steps,
        "BMI_category": bmi_category
    }])

    try:
        categorical_columns = ["Gender", "Occupation", "BMI_category", "Blood_Pressure"]

        for col in categorical_columns:
            user_data[col] = encoders[col].transform(user_data[col])

    except ValueError as e:
        return f"Input Error: {e}"

    numeric_prediction = xgb_model.predict(user_data)

    readable_prediction = target_encoder.inverse_transform(numeric_prediction)[0]

    return readable_prediction


if __name__ == "__main__":

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Unique y values:", y.unique())
    print("Target classes:", target_encoder.classes_)
    xgb_model.fit(X, y)

    print("Model trained successfully!")

    joblib.dump(xgb_model, "model.pkl")
    joblib.dump(encoders, "encoders.pkl")
    joblib.dump(target_encoder, "target_encoder.pkl")

    print("Model and encoders saved successfully!")