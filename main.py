from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from azure.cognitiveservices.vision.computervision import ComputerVisionClient 
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes, VisualFeatureTypes 
from msrest.authentication import CognitiveServicesCredentials 

import time 
import base64
import os
import requests 
from io import BytesIO
from PIL import Image 
from decouple import config 

AZURE_VISION_COG_KEY = config('AZURE_VISION_COGNITIVE_SERVICES_KEY')
AZURE_VISION_COG_ENDPOINT = config('AZURE_VISION_COGNITIVE_SERVICES_ENDPOINT')

# print(AZURE_COG_ENDPOINT)
computervision_client = ComputerVisionClient(AZURE_VISION_COG_ENDPOINT, CognitiveServicesCredentials(AZURE_VISION_COG_KEY))

app = FastAPI(title = "FastAPI + Azure Cognitive services.")

if not os.path.isdir("uploads"):
    os.mkdir("uploads")

UPLOADS_DIR = "uploads"

ANALYSE_ENDPOINT = AZURE_VISION_COG_ENDPOINT+"/vision/v3.1/analyze"
OCR_ENDPOINT = AZURE_VISION_COG_ENDPOINT+"/vision/v3.1/ocr"

headers = {
    'Content-Type': 'application/octet-stream',
    'Ocp-Apim-Subscription-Key': AZURE_VISION_COG_KEY
}

def image_analyse(image_path : str):
    params = {
        'language': 'en'
    }
    image_data = open(image_path, "rb").read()

    response = requests.post(
        ANALYSE_ENDPOINT,
        headers = headers,
        params = params,
        data = image_data
    )

    if response.status_code == 200:
        return response.json()
    else:
        return None

def image_ocr(image_path : str):
    params = {
        'language': 'en'
    }
    image_data = open(image_path, "rb").read()

    response = requests.post(
        OCR_ENDPOINT,
        headers = headers,
        params = params,
        data = image_data
    )

    sentences = []
    if response.status_code == 200:
        print(response.json())

        if not response.json()['regions']:
            return {
                "status" : False,
                "result" : "No text extracted from this Image."
            }
        
        for line in response.json()['regions'][0]['lines']:
            sentences.append(" ".join([word['text'] for word in line['words']]))

        return {
            "language" : response.json()['language'],
            "sentences" : sentences
        }

    else:
        return None 

@app.post("/image/{mode}", response_model=dict)
async def analyse_image(mode : str = "analyze", file : UploadFile = File(...)):
    extension = file.filename.split(".")[-1]
    if not extension.lower() in ["jpg", "png", "jpeg"]:
        raise HTTPException(
            detail = {
                "status" : False,
                "error" : "Only image files are supported i.e .jpg, .jpeg, .png"
            },
            status_code=422
    )

    print(mode)
    file_save_path = os.path.join(UPLOADS_DIR, file.filename)
    with open(file_save_path, "wb+") as f:
        f.write(await file.read())

    if mode == "analyze":
        result = image_analyse(file_save_path)
    elif mode == "ocr":
        result = image_ocr(file_save_path)

    return {
        "status" : True,
        "result" : result
    }
