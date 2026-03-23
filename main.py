from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from PIL import Image
import io
from predict import predict

# 1. FASTAPI APP
app = FastAPI(title="PFE Medical AI - TB Detection")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 2. ROUTES
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Lire l'image uploadée
    contents = await file.read()
    image    = Image.open(io.BytesIO(contents)).convert('RGB')

    # Prédiction + Grad-CAM
    result = predict(image)

    return result