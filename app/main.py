import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from source.constants import MODEL_PATH, CAT_FEATURES
from source.ml.data import process_data

app = FastAPI()


from pydantic import BaseModel


class Data(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float = Field(..., alias="capital-gain")
    capital_loss: float = Field(..., alias="capital-loss")
    hours_per_week: float = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")


model = joblib.load(MODEL_PATH / "model.joblib")
encoder = joblib.load(MODEL_PATH / "encoder.joblib")
lb = joblib.load(MODEL_PATH / "lb.joblib")


# Welcome user
@app.get("/")
def greetings():
    return {"Greetings": "User"}


@app.post("/inference/")
async def model_inference(data: Data):
    if data.age < 0:
        raise HTTPException(status_code=400, detail="age needs to be greater than 0.")
    
    if data.hours_per_week < 0:
        raise HTTPException(status_code=400, detail="hours_per_week needs to be greater than 0.")

    if hasattr(data, "model_dump"):  # Pydantic v2
        df_data = pd.DataFrame([data.model_dump(by_alias=True)])
    else:  # Pydantic v1
        df_data = pd.DataFrame([data.dict(by_alias=True)])
    
    # Process inference data with the process_data function.
    X_data, _, _, _ = process_data(
        df_data, categorical_features=CAT_FEATURES, encoder=encoder, lb=lb, training=False
    )

    prediction = model.predict(X_data)
    prediction = lb.inverse_transform(prediction)

    return {"prediction": prediction.tolist()}