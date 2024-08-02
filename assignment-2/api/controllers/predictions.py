from fastapi import HTTPException, Request
from model.model_loader import model_loader
import numpy as np


async def predict(
    injury: float,
    invage: float,
    passenger: float,
    speeding: float,
    truck: float,
    traffctl: str,
    light: str,
    alcohol: float,
    district: str,
    trsn: float,
    redlight: float,
):
    try:
        traffctl_no_control = 1.0 if traffctl == "no_control" else 0.0
        traffctl_automated = 1.0 if traffctl == "automated" else 0.0
        light_natural = 1.0 if light == "natural" else 0.0
        light_dark = 1.0 if light == "dark" else 0.0
        light_artificial = 1.0 if light == "artificial" else 0.0
        district_scarborough = 1.0 if district == "scarborough" else 0.0
        district_toronto = 1.0 if district == "toronto" else 0.0
        district_etobicoke = 1.0 if district == "etobicoke" else 0.0
        northYork = 1.0 if district == "northYork" else 0.0

        print( traffctl_no_control, traffctl_automated, light_natural, light_dark, light_artificial, district_scarborough, district_toronto, district_etobicoke, northYork)

        features = np.array(
            [
                [
                    injury,
                    invage,
                    passenger,
                    speeding,
                    truck,
                    traffctl_no_control,
                    light_natural,
                    light_dark,
                    alcohol,
                    traffctl_automated,
                    district_scarborough,
                    district_toronto,
                    district_etobicoke,
                    trsn,
                    redlight,
                    light_artificial,
                    northYork,
                ]
            ]
        )
        prediction = model_loader.predict(features)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
