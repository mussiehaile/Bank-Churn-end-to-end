import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import mlflow
import pickle

# Define the input data model
class InputData(BaseModel):
    credit_score: int
    age: int
    tenure: int
    balance: float
    products_number: int
    credit_card: int
    active_member: int
    estimated_salary: float


# with open(r"model/model.pkl","rb")as f:
#     model = pickle.load(f)

# Load the model.pkl file using the correct path
model_path = "/code/model/model.pkl"
model = joblib.load(model_path)

# Create the FastAPI app
app = FastAPI()

# Define a POST endpoint to make predictions
@app.post("/predict")
def predict(data: InputData):
    
    df = pd.DataFrame(data=[{
        "credit_score": data.credit_score,
        "age": data.age,
        "tenure": data.tenure,
        "balance": data.balance,
        "products_number": data.products_number,
        "credit_card": data.credit_card,
        "active_member": data.active_member,
        "estimated_salary": data.estimated_salary,
     }])


    # Make predictions using the loaded model
    predictions = model.predict(df)

    # Return the predictions as JSON response
    return {"predictions": predictions.tolist()}

# Run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
