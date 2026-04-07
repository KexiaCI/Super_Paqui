# Manual de Despliegue en Render y Configuración de Facebook Messenger

## 1. Crear el Servidor en Render
1. Sube esta carpeta (`messenger/`) a un repositorio en **GitHub**.
2. Entra a [Render.com](https://render.com) y crea una cuenta o inicia sesión.
3. Haz clic en **New +** y selecciona **Web Service**.
4. Conecta tu cuenta de GitHub y selecciona el repositorio que contiene tu backend de Messenger.
5. En la configuración del servicio de Render:
   - **Environment:** `Python 3`
   - **Build Command:** Usa `pip install -r requirements.txt`
   - **Start Command:** Usa `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Abre **Advanced** y añade tus Variables de Entorno (Environment Variables):
   - `FB_VERIFY_TOKEN`: El token secreto que inventes (ej: `micontrasenasecreta123`)
   - `FB_PAGE_ACCESS_TOKEN`: El token que te dará Facebook más adelante.
7. Haz clic en **Create Web Service**. Espera a que termine. Arriba verás una URL similar a `https://super-paqui-bot.onrender.com`.

## 2. Configurar el Webhook en Facebook Developers
1. Ve a [Meta for Developers](https://developers.facebook.com/).
2. Crea una aplicación tipo **Negocios** (Business) o selecciona tu app existente.
3. Agrega el producto **Messenger** a tu app.
4. En la sección *Tokens de Acceso*, vincula tu página de Facebook "Super Paqui" y **Genera un Token** (`PAGE_ACCESS_TOKEN`). Copia este token y ponlo en Render.
5. En la sección *Webhooks*, haz clic en **Configurar Webhook**.
   - **URL de devolución de llamada:** Usa tu URL de Render + `/webhook` (ej: `https://super-paqui-bot.onrender.com/webhook`)
   - **Token de verificación:** Pon el texto exacto que pusiste en `FB_VERIFY_TOKEN` en Render.
   - Haz clic en Verificar y Guardar. Si Render está encendido, lo aceptará.
6. En Webhooks de la página, asegúrate de **suscribir** la página a la app y marca la casilla `messages` (y `messaging_postbacks` si usas botones).

## 3. Política de Privacidad
Para que tu App pase a estado **Público** ("En vivo"), Facebook te requerirá una URL de Política de Privacidad.
En tu dashboard de Meta Developers > Configuración > Básica:
- Pon la ruta de tu Política de privacidad. Como lo embebimos en la App, tu enlace será: `https://TU-URL-DE-RENDER.onrender.com/privacidad`
- Hemos preparado esta página que cumple con todas las cláusulas requeridas por Facebook.

## 4. Probando el Bot
Abre con una cuenta que sea Administradora o Verificadora de la app, manda un mensaje a la Página por Messenger y ¡listo! Empezará a chatear y mandar cotizaciones.
