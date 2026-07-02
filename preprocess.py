import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess():
    df = pd.read_csv("sleep_disorder_data_advanced_2000 (1).csv")

    df["Sleep_disorder"] = df["Sleep_disorder"].fillna("None")

    encoders = {}

    categorical_columns = [
        "Gender",
        "Occupation",
        "BMI_category"
    ]

    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    target_encoder = LabelEncoder()
    df["Sleep_disorder"] = target_encoder.fit_transform(
        df["Sleep_disorder"]
    )

    X = df.drop("Sleep_disorder", axis=1)
    y = df["Sleep_disorder"]

    print(df["Sleep_disorder"].value_counts())
    print("Unique classes:", df["Sleep_disorder"].unique())
    print("Number of classes:", len(df["Sleep_disorder"].unique()))
    return X, y, encoders, target_encoder