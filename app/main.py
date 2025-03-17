from fastapi import FastAPI, File, UploadFile, Query
import shutil
import pytesseract
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import torch

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

@app.get("/hello")
def start():
    return {
        "Hello world"
    }
@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...), embed: bool = Query(True, title="use embed or not")):
    if embed:
        #Sentences we want sentence embeddings for
        sentences = ['Привет! Как твои дела?',
                    'А правда, что 42 твое любимое число?']

        #Tokenize sentences
        encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=24, return_tensors='pt')

        #Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        #Perform pooling. In this case, mean pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])        
        return sentence_embeddings

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

#curl -X POST "http://localhost:80/upload-image/?embed=true" -F "file=@C:\proger\sticker_index_api\image.png