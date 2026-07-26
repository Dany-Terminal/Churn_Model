import joblib
import pandas as pd


model = joblib.load("../Model/churn_model.pkl")


customer = pd.DataFrame(
    {
        "Age": [35],
        "Gender": ["Male"],
        "Country": ["India"],
        "Membership_Years": [4],
        "Cart_Abandonment_Rate": [30],
        "Days_Since_Last_Purchase": [60],
        "Customer_Service_Calls": [5],
        "Email_Open_Rate": [20],
        "Lifetime_Value": [5000],
    }
)


probability = model.predict_proba(customer)[:, 1]


prediction = (probability > 0.35).astype(int)


print(probability)
print(prediction)
