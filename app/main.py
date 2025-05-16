from fastapi import FastAPI, File, UploadFile, Query
import shutil
import pytesseract
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import torch
from pydantic import BaseModel
import os

# Указываем путь до исполняемого файла Tesseract (уже настроено в Docker)
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


app = FastAPI()

#Load AutoModel from huggingface model repository
tokenizer = AutoTokenizer.from_pretrained("ai-forever/sbert_large_nlu_ru")
model = AutoModel.from_pretrained("ai-forever/sbert_large_nlu_ru")

class TextRecognitionResponse(BaseModel):
    recognized_text: str

@app.get("/hello")
def start():
    return {
        "Hello world"
    }

@app.post("/upload-image/", response_model=TextRecognitionResponse)
async def upload_image(file: UploadFile = File(...)) -> dict:
    """
    Обработчик для загрузки изображения и распознавания текста с использованием Tesseract OCR.
    
    :param file: Загруженный файл с изображением
    :return: Словарь с распознанным текстом
    """
    
    try:
        # Создаем временный файл для хранения загруженного изображения
        temp_file_path = f"/tmp/{os.path.basename(file.filename)}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(1)    
        # Открываем изображение и распознаём текст
        img = Image.open(temp_file_path)
        text = pytesseract.image_to_string(img)
        print(2)    

        # Удаляем временное изображение
        os.remove(temp_file_path)
        
        sentences = text.strip()
        encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=24, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        #Perform pooling. In this case, mean pooling
        print(3)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])        
        print(4)
        return {"recognized_text": text.strip(),  'embedding': sentence_embeddings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке изображения: {str(e)}")



#curl -X POST "http://localhost:80/upload-image/?embed=false" -F "file=@C:\projects\sticker_index_api\image.png"