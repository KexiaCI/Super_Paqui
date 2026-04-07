# Despliegue del WhatsApp Bot en Hugging Face Spaces (Docker)

Hugging Face Spaces es una excelente plataforma para alojar tu bot de WhatsApp, pero dado que utilizamos **Selenium y Google Chrome** para enlazar WhatsApp Web, el entorno base de Python de Hugging Face no incluye los navegadores por defecto. 

Para solucionar esto, necesitas configurar el Space usando **Docker**. Esto nos permite instalar Chrome, y todas las dependencias requeridas en un contenedor virtualizado.

## 1. Crear el Space en Hugging Face

1. Ve a tu cuenta de [Hugging Face](https://huggingface.co/) y navega a **Spaces**.
2. Haz clic en **Create new Space**.
3. Ingresa un nombre para tu Space (por ejemplo: `super-paqui-whatsapp`).
4. Selecciona **Docker** como el Space SDK. 
   *(Esta es la plantilla "Blank" para Docker)*.
5. Haz clic en **Create Space**.

## 2. Archivos Necesarios

Debes subir los archivos de tu bot (todo lo que está dentro de la carpeta `whatsapp` localmente) al repositorio de Hugging Face. Puedes copiarlos manualmente en la pestaña de Archivos o conectándote vía Git.

### A) Dockerfile (Clave para instalar Chrome)

Crea un archivo llamado `Dockerfile` exacto dentro del Space con el siguiente contenido:

```dockerfile
FROM python:3.10-slim

# Evitar prompts interactivos
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema y Google Chrome
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip \
    curl \
    && curl -fsSL https://dl-ssl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/googlechrome-linux-keyring.gpg \
    && sh -c 'echo "deb [arch=amd64 signed-by=/usr/share/keyrings/googlechrome-linux-keyring.gpg] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list' \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Configurar el usuario para Hugging Face Spaces (Opcional pero Recomendado para evitar problemas con root)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copiar el requirements.txt e instalar dependencias
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente completo
COPY --chown=user . .

# Comando de inicio del servidor FastAPI con Uvicorn
# (Asegúrate de que corre en el puerto 7860 y en 0.0.0.0 para Hugging Face)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

### B) Tu código fuente y Configuración de Selenium

Asegúrate de incluir en la carpeta el `main.py`, `chatbot.py`, `whatsapp_bridge.py`, tus archivos de credenciales `.json`, y el `requirements.txt`.

Dado que Docker se ejecuta sin interfaz gráfica, confirma que tu automatización de Selenium (usualmente en `whatsapp_bridge.py`) tenga activos los comandos para el modo fantasma (headless):

```python
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

chrome_options = Options()
chrome_options.add_argument("--headless=new") # IMPORTANTE: Chrome debe ir oculto
chrome_options.add_argument("--no-sandbox")   # IMPORTANTE: Requerido en Docker
chrome_options.add_argument("--disable-dev-shm-usage") # IMPORTANTE: Evita crash por memoria

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)
```

## 3. Obtención del Código QR

El bot en tu computadora abría una ventana visual con el QR. 
En Hugging Face **no habrá ventana**. Por lo tanto:

- Debes estar atento a la pestaña **Logs** de tu Space en Hugging Face.
- Si previamente implementaste terminal logging u obtención del QR como link en `whatsapp_bridge.py`, el bot te arrojará el código en la consola del log.
- Deberás escanearlo utilizando tu celular viendo la pantalla de Hugging Face.
- **ALERTA HUGGING FACE**: Cuando el Space entra en modo "Sleeping" (inactividad) y vuelve a despertar, *la sesión del navegador se perderá* debido a lo efímero de los contenedores Docker básicos. Te pedirá escanear de nuevo el código. 
- *Para evitar esto:* Puedes contratar el "Persistent Storage" en Settings del Space de Hugging Face (cuesta pocos dólares o configurarlo en los Free Tiers) para que guarde tu `user-data-dir` (la carpeta de perfil de chrome) de forma permanente.

## 4. Variables de Entorno (Opcional)

Si utilizas llaves API, tokens o configuraciones secretas, dirígete a los **Settings** -> **Variables and secrets** de tu Space y dálos de alta sin subirlos directamente en el código de GitHub para mayor seguridad.

¡Una vez todo arriba y en la consola indique "Running", tu WhatsApp bot estará sirviendo desde la nube!
