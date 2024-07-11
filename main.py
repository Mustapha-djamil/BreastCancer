from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Fonction pour charger le modèle depuis les fichiers JSON et H5
def load_saved_model(model_json_path, model_weights_path):
    json_file = open(model_json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_weights_path)
    loaded_model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    return loaded_model

# Charger le modèle depuis les fichiers JSON et H5
model = load_saved_model('model.json', 'model.h5')

# Fonction pour prédire à partir de l'image téléchargée
def predict_image(img_file_path, model):
    img = load_img(img_file_path, target_size=(25, 25))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    has_no_cancer = '% NO cancer: ' + str(round(prediction[0][1]*100, 2)) + "%"
    has_cancer = '% cancer: ' + str(round(prediction[0][0]*100, 2)) + '%'
    return has_cancer, has_no_cancer

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    filename = os.path.join('uploads', file.filename)
    with open(filename, 'wb') as f:
        f.write(contents)
    has_no_cancer, has_cancer = predict_image(filename, model)
    os.remove(filename)  # Supprimer l'image temporaire
    return templates.TemplateResponse("result.html", {"request": request, "has_no_cancer": has_no_cancer, "has_cancer": has_cancer})
