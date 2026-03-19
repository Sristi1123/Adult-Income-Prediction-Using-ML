from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

class PersonData(BaseModel):
    age: int
    workclass: str
    education_num: int
    marital_status: str
    occupation: str
    sex: str
    hours_per_week: int

@app.post("/predict")
async def predict_income(person: PersonData):
    # Load model and make prediction
    model = joblib.load('best_model.pkl')
    # ... preprocessing and prediction logic
    return {"prediction": result}