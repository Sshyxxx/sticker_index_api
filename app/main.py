from fastapi import FastAPI, File, UploadFile
import shutil
import pytesseract
from PIL import Image

# Указываем путь до исполняемого файла Tesseract (уже настроено в Docker)
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'



app = FastAPI()

@app.get("/hello")
def start():
    return {
        "Hello world"
    }

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        with open(f"{file.filename}", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            # Открываем изображение
            img = Image.open(file.filename)  #ваше изображение
            # Распознаем текст на изображении
            text = pytesseract.image_to_string(img)

        return {"filename":text}
    finally:
        file.file.close()

#curl -X POST "http://localhost:80/upload-image/" -F "file=@C:\proger\sticker_index_api\image.png