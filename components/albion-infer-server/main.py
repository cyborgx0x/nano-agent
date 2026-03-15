from fastapi import FastAPI
from detection import game_detection, get_text
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel


app = FastAPI()
data = {}


@app.post("/detect/")
async def detection(file: UploadFile):
    im = Image.open(file.file)
    result = get_text(image=im)
    return result

@app.post("/fiber_detection/")
async def fiber_detection(file: UploadFile):
    im = Image.open(file.file)
    return game_detection(image=im)
    

class Log(BaseModel):
   id: int
   status: str

@app.post("/status")
def add_book(log: Log):
   data[log.id] = log.status
   return data

@app.get("/status/{id}")
def get_books(id:int):
   return data.get(id)

