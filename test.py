import pandas as pd

df = pd.read_csv("Sleep_Data_Sampled.csv")

print("120/90" in df["Blood Pressure"].values)
print(df["Blood Pressure"].unique())