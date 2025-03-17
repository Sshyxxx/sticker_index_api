FROM python:3.9-slim

#
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget gnupg lsb-release ca-certificates && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr && \
    apt-get clean

#
RUN pip install --no-cache-dir --upgrade pip
RUN pip install pillow pytesseract

#
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

#
COPY ./app /code/app

#
WORKDIR /code/app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]