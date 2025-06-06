import logging
from typing import List
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
import shutil
import pytesseract
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import torch
from pydantic import BaseModel
import os

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Указываем путь до исполняемого файла Tesseract (уже настроено в Docker)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


app = FastAPI()

logger.info("Loading tokenizer from HuggingFace...")
tokenizer = AutoTokenizer.from_pretrained("ai-forever/sbert_large_nlu_ru", verbose=True)
logger.info("Tokenizer loaded.")
logger.info("Loading model from HuggingFace...")
model = AutoModel.from_pretrained("ai-forever/sbert_large_nlu_ru")
logger.info("Model loaded successfully.")


class TextRecognitionResponse(BaseModel):
    recognized_text: str
    embedding: List


@app.get("/hello")
def start():
    logger.info("GET /hello called")
    return {"Hello world"}


@app.post("/upload-image/", response_model=TextRecognitionResponse)
async def upload_image(file: UploadFile = File(...)) -> dict:
    """
    Обработчик для загрузки изображения и распознавания текста с использованием Tesseract OCR.

    :param file: Загруженный файл с изображением
    :return: Словарь с распознанным текстом
    """
    logger.info(f"Received file upload: {file.filename}")
    try:
        # Создаем временный файл для хранения загруженного изображения
        temp_file_path = f"/tmp/{os.path.basename(file.filename)}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Saved uploaded file to {temp_file_path}")

        # Открываем изображение и распознаём текст
        img = Image.open(temp_file_path)
        text = pytesseract.image_to_string(img)
        logger.info(f"Text recognized: {text.strip()}")

        # Удаляем временное изображение
        os.remove(temp_file_path)
        logger.info(f"Temporary file {temp_file_path} deleted")

        sentences = text.strip()
        encoded_input = tokenizer(
            sentences, padding=True, truncation=True, max_length=24, return_tensors="pt"
        )
        with torch.no_grad():
            model_output = model(**encoded_input)
        # Perform pooling. In this case, mean pooling
        sentence_embeddings = mean_pooling(
            model_output, encoded_input["attention_mask"]
        )
        logger.info("Embedding calculated successfully.")

        return {"recognized_text": text, "embedding": sentence_embeddings.tolist()}

#        return {"recognized_text": text.strip(), "embedding": sentence_embeddings}
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Ошибка при обработке изображения: {str(e)}"
        )

@app.post("/text-embedding/")
async def text_embedding(text: str) -> dict:

        sentences = text.strip()
        encoded_input = tokenizer(
            sentences, padding=True, truncation=True, max_length=24, return_tensors="pt"
        )
        with torch.no_grad():
            model_output = model(**encoded_input)
        # Perform pooling. In this case, mean pooling
        sentence_embeddings = mean_pooling(
            model_output, encoded_input["attention_mask"]
        )
        logger.info("Embedding calculated successfully.")

        return {"embedding": sentence_embeddings.tolist()}

@app.post("/search/")
async def handle_search_request(req: SearchRequest):
    """
    Обработчик маршрута /search/, очищает текст от команды и сохраняет очищенное значение.
    """
    # Извлечение текста после команды /search
    match = re.match(r"/search\s+(.*)", req.command)
    if not match:
        raise HTTPException(status_code=400, detail="Invalid format. Expected '/search <text>'")

    cleaned_text = match.group(1)

    # Сохраняем очищенный текст для дальнейшей обработки
    indexing_results.append(cleaned_text)

    return {"message": f"Indexed '{cleaned_text}'"}
