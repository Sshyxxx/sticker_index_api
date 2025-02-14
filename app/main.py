from fastapi import FastAPI, File, UploadFile
import shutil

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
        return {"filename": file.filename}
    finally:
        file.file.close()

#curl -X POST "http://localhost:80/upload-image/" -F "file=@C:\proger\sticker_index_api\image.png