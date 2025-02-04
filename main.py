from fastapi import FastAPI
from fastapi.responses import FileResponse
 
sticker_index_api = FastAPI()
 
@sticker_index_api.get("/ocr_get_text")
def root():
    return {"message": "text"}

@sticker_index_api.get("/")
def root():
    return FileResponse("index.html")
 
# альтернативный вариант
@sticker_index_api.get("/file", response_class = FileResponse)
def root_html():
    return "index.html"

