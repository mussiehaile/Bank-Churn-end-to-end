import pandas as pd
import numpy as np



df = pd.read_csv('/home/mussie/Music/home projects/final proj/Bank-Churn-end-to-end/src/data/Bank Customer Churn Prediction.csv')

train_data = df[:8000]
current_data = df[8000:]

train_data.to_csv("train_data.csv",index=False)
current_data.to_csv("current_data.csv" , index=False)