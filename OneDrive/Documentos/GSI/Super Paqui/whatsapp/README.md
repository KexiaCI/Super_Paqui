---
title: WhatsApp Chatbot San Diego
emoji: 🚚
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

# WhatsApp Chatbot - Paquetería San Diego

Este es el backend del chatbot de WhatsApp para Paquetería San Diego, desplegado en Hugging Face Spaces.

## Configuración

- **SDK**: Docker
- **Puerto**: 7860
- **Framework**: FastAPI

## Archivos Necesarios

- `main.py`: Punto de entrada de la aplicación.
- `chatbot.py`: Lógica del chatbot.
- `intents.json`: Base de conocimiento del chatbot.
- `requirements.txt`: Dependencias de Python.
- `Dockerfile`: Configuración del contenedor.
- `RORS890824N37.json`: Credenciales de Google Sheets.
- `static/` y `templates/`: Archivos para la interfaz web.

## Cómo usar

Una vez desplegado, la interfaz web estará disponible en la URL del Space. El endpoint `/chat` recibe mensajes y devuelve respuestas basadas en los intents configurados.
