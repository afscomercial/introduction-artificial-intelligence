from fastapi import HTTPException, Request
from model.model_loader import model_loader
import numpy as np

# from jose import JWTError

PREDICTIONS = [
    {
        "id": "1",
        "type": "two_bedroom",
        "city": "toronto",
        "prediction": "2400",
        "complete": False,
    },
    {
        "id": "2",
        "type": "room",
        "city": "mississauga",
        "prediction": "1000",
        "complete": False,
    },
    {
        "id": "3",
        "type": "room",
        "city": "brampton",
        "prediction": "1500",
        "complete": False,
    },
    {
        "id": "4",
        "type": "studio",
        "city": "scarborough",
        "prediction": "1800",
        "complete": False,
    },
    {
        "id": "5",
        "type": "one_bedroom",
        "city": "toronto",
        "prediction": "2200",
        "complete": False,
    },
]

async def read_prediction(prediction_id: str):
    for prediction in PREDICTIONS:
        if prediction.get("id") == prediction_id:
            return prediction
    raise HTTPException(status_code=404, detail="Prediction not found")


async def predict():
    try:
        features = np.array(
            [
                [
                    3.0,
                    2.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ]
        )
        predictions = model_loader.predict(features)
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
