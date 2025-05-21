from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load artifacts
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('feature_names.pkl')
metrics = joblib.load('metrics.pkl')

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve HTML
@app.get("/", response_class=HTMLResponse)
def home():
    return open("templates/form.html").read()

@app.get("/result", response_class=HTMLResponse)
def result():
    return open("templates/result.html").read()

# Input schema
class LoanInput(BaseModel):
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Dependents: int
    Gender_Male: int
    Married_Yes: int
    Education_Not_Graduate: int = 0
    Self_Employed_Yes: int
    Property_Area_Semiurban: int
    Property_Area_Urban: int

@app.post("/predict")
async def predict(data: LoanInput):
    sample = data.dict()

    df = pd.DataFrame([sample])
    for col in columns:
        if col not in df.columns:
            df[col] = 0
    df = df[columns]

    X_scaled = scaler.transform(df)
    pred = int(model.predict(X_scaled)[0])

    return JSONResponse({
        "prediction": pred,
        "metrics": {
            "accuracy": f"{metrics['accuracy'] * 100:.2f}%",
            "precision": f"{metrics['precision'] * 100:.2f}%",
            "recall": f"{metrics['recall'] * 100:.2f}%",
            "f1": f"{metrics['f1'] * 100:.2f}%"
        }
    })
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
