import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# 1. TRAINING PIPELINE

# Load data
df = pd.read_csv('sleep_disorder_data_advanced_2000 (1).csv')
df['Sleep_disorder'] = df['Sleep_disorder'].fillna('None')

X = df.drop(columns=['Sleep_disorder'])
y = df['Sleep_disorder']

# Create a dictionary to store our encoders
# We MUST keep these to encode the user's input later
encoders = {}

# Encode categorical features and save the encoders
cat_cols = X.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Encode target variable
target_le = LabelEncoder()
y_encoded = target_le.fit_transform(y)

# Train the XGBoost model on the full dataset
xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',
    eval_metric='mlogloss',
    random_state=42
)
xgb_model.fit(X, y_encoded)
print("Model trained successfully!\n")


# 2. INFERENCE FUNCTION (User Input)

def predict_sleep_disorder(age, gender, occupation, sleep_duration, 
                           quality_of_sleep, physical_activity, stress_level, 
                           bmi, heart_rate, daily_steps, sleep_efficiency, bmi_category):
    
    # 1. Create a DataFrame from the user's input
    user_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Occupation': occupation,
        'Sleep_duration': sleep_duration,
        'Quality_of_sleep': quality_of_sleep,
        'Physical_activity': physical_activity,
        'Stress_Level': stress_level,
        'BMI': bmi,
        'Heart_rate': heart_rate,
        'Daily_steps': daily_steps,
        'Sleep_Efficiency': sleep_efficiency,
        'BMI_category': bmi_category
    }])
    
    # 2. Apply the saved encoders to the user's text inputs
    try:
        for col in cat_cols:
            # We use transform(), NOT fit_transform(), to apply the exact same mapping
            user_data[col] = encoders[col].transform(user_data[col])
    except ValueError as e:
        return f"Input Error: Unrecognized category. Please check your spelling for Gender, Occupation, or BMI_category. Details: {e}"

    # 3. Make the prediction
    numeric_prediction = xgb_model.predict(user_data)
    
    # 4. Decode the numeric prediction back into a readable string ('None', 'Insomnia', 'Sleep Apnea')
    readable_prediction = target_le.inverse_transform(numeric_prediction)[0]
    
    return readable_prediction


# 3. TEST THE FUNCTION

print("--- Testing User Input ---")

# 3. INTERACTIVE USER INPUT & PREDICTION

print("\n--- Enter Your Details for Sleep Disorder Prediction ---")

try:
    # Collecting and casting user inputs
    user_age = int(input("Enter Age (e.g., 42): "))
    user_gender = input("Enter Gender (e.g., Male/Female): ")
    user_occupation = input("Enter Occupation (e.g., Student, Doctor): ")
    user_sleep_duration = float(input("Enter Sleep Duration in hours (e.g., 6.3): "))
    user_quality_of_sleep = int(input("Enter Quality of Sleep scale 1-10 (e.g., 9): "))
    user_physical_activity = float(input("Enter Physical Activity level (e.g., 2.1): "))
    user_stress_level = int(input("Enter Stress Level scale 1-10 (e.g., 2): "))
    user_bmi = float(input("Enter BMI (e.g., 23.3): "))
    user_heart_rate = int(input("Enter Heart Rate bpm (e.g., 83): "))
    user_daily_steps = int(input("Enter Daily Steps (e.g., 14333): "))
    user_sleep_efficiency = float(input("Enter Sleep Efficiency percentage (e.g., 99.2): "))
    user_bmi_category = input("Enter BMI Category (e.g., Normal, Overweight): ")

    # Passing the custom inputs into the function
    predicted_disorder = predict_sleep_disorder(
        age=user_age,
        gender=user_gender,
        occupation=user_occupation,
        sleep_duration=user_sleep_duration,
        quality_of_sleep=user_quality_of_sleep,
        physical_activity=user_physical_activity,
        stress_level=user_stress_level,
        bmi=user_bmi,
        heart_rate=user_heart_rate,
        daily_steps=user_daily_steps,
        sleep_efficiency=user_sleep_efficiency,
        bmi_category=user_bmi_category
    )

    print(f"\n>>> Based on your inputs, the model predicts: {predicted_disorder} <<<")

except ValueError as e:
    print(f"\nInput Error: Please ensure you enter numbers for numeric fields like Age, BMI, and Steps. Details: {e}")