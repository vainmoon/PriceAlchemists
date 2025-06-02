FROM python:3.9


WORKDIR /backend


COPY ./requirements.txt /backend/requirements.txt


RUN pip install --no-cache-dir --upgrade -r /backend/requirements.txt

COPY ./app /backend/app

EXPOSE 8000

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]