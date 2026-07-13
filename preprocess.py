import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess():
    df = pd.read_csv("Sleep_Data_Sampled.csv")

    df = df.rename(columns={
        "Sleep Duration": "Sleep_duration",
        "Quality of Sleep": "Quality_of_sleep",
        "Physical Activity Level": "Physical_activity",
        "Stress Level": "Stress_Level",
        "BMI Category": "BMI_category",
        "Heart Rate": "Heart_rate",
        "Daily Steps": "Daily_steps",
        "Sleep Disorder": "Sleep_disorder",
        "Blood Pressure": "Blood_Pressure"
    })

    df["Sleep_disorder"] = df["Sleep_disorder"].fillna("None")

    encoders = {}

    categorical_columns = [
        "Gender",
        "Occupation",
        "BMI_category",
        "Blood_Pressure"
    ]

    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    target_encoder = LabelEncoder()
    df["Sleep_disorder"] = target_encoder.fit_transform(
        df["Sleep_disorder"]
    )

    X = df.drop(["Person ID", "Sleep_disorder"], axis=1)
    X = X[
    [
        "Age",
        "Gender",
        "Occupation",
        "Sleep_duration",
        "Quality_of_sleep",
        "Physical_activity",
        "Stress_Level",
        "Blood_Pressure",
        "Heart_rate",
        "Daily_steps",
        "BMI_category"
    ]
]
    y = df["Sleep_disorder"]

    print(df["Sleep_disorder"].value_counts())
    print("Unique classes:", df["Sleep_disorder"].unique())
    print("Number of classes:", len(df["Sleep_disorder"].unique()))
    return X, y, encoders, target_encoder