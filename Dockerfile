FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias necesarias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código
COPY . .

# Comando de ejecución
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
