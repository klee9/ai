FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

COPY . /code

ENV PORT=7860
EXPOSE 7860

CMD ["bash", "-lc", "uvicorn app.api:app --host 0.0.0.0 --port ${PORT}"]
