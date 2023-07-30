import pandas as pd
import numpy as np



df = pd.read_csv('/home/mussie/Music/home projects/final proj/Bank-Churn-end-to-end/src/data/Bank Customer Churn Prediction.csv')

reference_data = df[:8000]
current_data = df[8000:]

reference_data.to_csv("reference_data.csv",index=False)
current_data.to_csv("current_data.csv" , index=False)