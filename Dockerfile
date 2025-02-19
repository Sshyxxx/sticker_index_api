FROM debian:bullseye-slim
# 
FROM python:3.9

# 
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt

RUN apt-get update && \
    apt-get install -y wget gnupg lsb-release ca-certificates && \
    apt-get install -y tesseract-ocr && \
    apt-get clean

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN pip install pillow
RUN pip install pytesseract


# 
COPY ./app /code/app

# 
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]