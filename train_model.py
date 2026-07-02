import pandas as pd
import xgboost as xgb
import warnings
import joblib
from preprocess import preprocess

warnings.filterwarnings("ignore")

X, y, encoders, target_encoder = preprocess()

xgb_model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=len(target_encoder.classes_),
    eval_metric="mlogloss",
    random_state=42
)

def predict_sleep_disorder(age, gender, occupation, sleep_duration,
                           quality_of_sleep, physical_activity, stress_level,
                           bmi, heart_rate, daily_steps, sleep_efficiency,
                           bmi_category):

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

    try:
        categorical_columns = ["Gender", "Occupation", "BMI_category"]

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