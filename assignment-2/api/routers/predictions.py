from fastapi import Body, APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from controllers.predictions import predict


router = APIRouter()


templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "home.html", {"request": request, "prediction": None}
    )


@router.post("/predictions", response_class=HTMLResponse)
async def predictions(
    request: Request,
    injury: str = Form(...),
    invage: str = Form(...),
    passenger: str = Form(...),
    speeding: str = Form(...),
    truck: str = Form(...),
    traffctl: str = Form(...),
    lightNatural: str = Form(...),
    lightDark: str = Form(...),
    alcohol: str = Form(...),
    traffctlAutomated: str = Form(...),
    scarborough: str = Form(...),
    toronto: str = Form(...),
    etobicoke: str = Form(...),
    trsnCity: str = Form(...),
    redlight: str = Form(...),
    lightArtificial: str = Form(...),
    northYork: str = Form(...),
):
    prediction = await predict(
        injury,
        invage,
        passenger,
        speeding,
        truck,
        traffctl,
        lightNatural,
        lightDark,
        alcohol,
        traffctlAutomated,
        scarborough,
        toronto,
        etobicoke,
        trsnCity,
        redlight,
        lightArtificial,
        northYork,
    )
    return templates.TemplateResponse(
        "home.html", {"request": request, "prediction": prediction}
    )
