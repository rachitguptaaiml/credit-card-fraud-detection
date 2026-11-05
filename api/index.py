
import pandas as pd
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

app = FastAPI(title="Credit Card Fraud Detection API")

# Generate demo data
np.random.seed(42)

df = pd.DataFrame({
    "Amount": np.random.uniform(10, 1000, 10000),
    "Time": np.random.uniform(1, 1440, 10000),
    "Location": np.random.choice(
        ["USA", "Canada", "UK"], 10000
    ),
    "Card Type": np.random.choice(
        ["Visa", "Mastercard", "Amex"], 10000
    ),
    "Fraud": np.random.choice([0, 1], 10000)
})

X = df.drop("Fraud", axis=1)
y = df["Fraud"]

# Process numerical and categorical features correctly
preprocessor = ColumnTransformer([
    ("categorical",
     OneHotEncoder(handle_unknown="ignore"),
     ["Location", "Card Type"])
], remainder="passthrough")

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)


class Transaction(BaseModel):
    Amount: float
    Time: float
    Location: str
    card_type: str


@app.get("/")
def home():
    return {
        "message": "Credit Card Fraud Detection API is running"
    }


@app.post("/predict")
def predict(transaction: Transaction):
    data = pd.DataFrame([{
        "Amount": transaction.Amount,
        "Time": transaction.Time,
        "Location": transaction.Location,
        "Card Type": transaction.card_type
    }])

    prediction = int(model.predict(data)[0])

    return {
        "prediction": prediction,
        "result": "Fraud" if prediction == 1 else "Legitimate"
    }
