# syntax=docker/dockerfile:1.4

FROM python:3.12

WORKDIR /app
COPY . .

RUN pip3 install --upgrade pip
RUN pip3 install uv
RUN uv pip install -r --system requirements.txt

ENTRYPOINT ["python3"]
CMD ["bot.py"]