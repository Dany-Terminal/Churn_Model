from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

from pathlib import Path

model_path = Path("Model/churn_model.pkl").resolve()
# Load trained pipeline
model = joblib.load(model_path)

app = FastAPI(title="Customer Churn Prediction API")


# Define input structure
class Customer(BaseModel):
    Age: float
    Gender: str
    Country: str
    Membership_Years: float
    Cart_Abandonment_Rate: float
    Days_Since_Last_Purchase: float
    Customer_Service_Calls: float
    Email_Open_Rate: float
    Lifetime_Value: float


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}


@app.post("/predict")
def predict(customer: Customer):

    # Convert input into dataframe
    customer_df = pd.DataFrame([customer.model_dump()])

    # Get churn probability
    churn_probability = model.predict_proba(customer_df)[0][1]

    # Your chosen threshold from Model C
    threshold = 0.35

    churn_prediction = int(churn_probability >= threshold)

    return {
        "churn_probability": round(float(churn_probability), 4),
        "churn_prediction": churn_prediction,
        "message": "Customer likely to churn"
        if churn_prediction == 1
        else "Customer likely to stay",
    }
