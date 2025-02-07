from fastapi import FastAPI, Query
import requests
from PIL import Image
from io import BytesIO

app = FastAPI()

@app.get("/items/{item_id}")
def process_url_image(item_id: int, image_url: str = Query(None), q: str | None = None):
    if image_url is not None:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        print(img.size)
    
    return {
        "item_id": item_id,
        "image_url": image_url,
        "q": q
    }

#https://i1.sndcdn.com/avatars-000581659953-k4k5gk-t500x500.jpg
#curl -Uri GET http://localhost:80/items/123?image_url=https://i1.sndcdn.com/avatars-000581659953-k4k5gk-t500x500.jpg&q=example_query