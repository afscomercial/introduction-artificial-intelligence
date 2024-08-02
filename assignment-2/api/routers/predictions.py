import os
from fastapi import Body, APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from controllers.predictions import predict
from dotenv import load_dotenv


load_dotenv()
router = APIRouter()
current_directory = os.path.dirname(os.path.abspath(__file__))

templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "home.html", {"request": request, "prediction": None, "path": os.getenv('MODEL_PATH'), "current_directory": current_directory}
    )


@router.post("/predictions", response_class=HTMLResponse)
async def predictions(
    request: Request,
    injury: float = Form(...),
    invage: float = Form(...),
    passenger: float = Form(...),
    speeding: float = Form(...),
    truck: float = Form(...),
    traffctl: str = Form(...),
    light: str = Form(...),
    alcohol: float = Form(...),
    district: str = Form(...),
    trsn: float = Form(...),
    redlight: float = Form(...),
):
    prediction = await predict(
       injury,
        invage,
        passenger,
        speeding,
        truck,
        traffctl,
        light,
        alcohol,
        district,
        trsn,
        redlight,
    )
    return templates.TemplateResponse(
        "home.html", {"request": request, "prediction": prediction, "path": os.getenv('MODEL_PATH'), "current_directory": current_directory}
    )
