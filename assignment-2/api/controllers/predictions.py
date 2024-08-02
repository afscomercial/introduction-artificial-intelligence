from fastapi import HTTPException, Request
from model.model_loader import model_loader
import numpy as np



async def predict(
    injury: float,
    invage: float,
    passenger: float,
    speeding: float,
    truck: float,
    traffctl: float,
    lightNatural: float,
    lightDark: float,
    alcohol: float,
    traffctlAutomated: float,
    scarborough: float,
    toronto: float,
    etobicoke: float,
    trsnCity: float,
    redlight: float,
    lightArtificial: float,
    northYork: float,
):
    try:
        features = np.array(
            [
                [
                    float(injury),
                    float(invage),
                    float(passenger),
                    float(speeding == "Yes"),
                    float(truck == "Yes"),
                    float(traffctl == "Yes"),
                    float(lightNatural == "Yes"),
                    float(lightDark == "Yes"),
                    float(alcohol == "Yes"),
                    float(traffctlAutomated == "Yes"),
                    float(scarborough == "Yes"),
                    float(toronto == "Yes"),
                    float(etobicoke == "Yes"),
                    float(trsnCity == "Yes"),
                    float(redlight == "Yes"),
                    float(lightArtificial == "Yes"),
                    float(northYork == "Yes"),
                ]
            ]
        )
        prediction = model_loader.predict(features)[0]
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
