from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
import uvicorn
import os

# ─── Inicialización lazy del chatbot para evitar lentitud en startup ──────────
chatbot_module = None
import json

def get_chatbot():
    global chatbot_module
    if chatbot_module is None:
        import chatbot as cb
        chatbot_module = cb
    return chatbot_module

# ─── Lógica de precios (del Cuestionario) ─────────────────────────────────────
PRICES = {
    "GA_AL": {
        "GTO": {
            "18gal": 90, "20gal": 100, "22gal": 120, "27gal": 130,
            "30gal": 140, "35gal": 160, "40gal": 170, "45gal": 180,
            "50gal": 200, "55gal": 220, "57gal": 230, "60gal": 240,
            "65gal": 260, "70gal": 280, "75gal": 300, "77gal": 310,
            "HD small": 90, "HD Med-Uhaul Mediana": 120, "HD Large": 140, "HD XL": 180,
            "Caja 18*18*24": 140, "Caja 22*22*22": 180, "Caja 24*24*24": 220,
            "Caja 24*24*26": 240, "Caja 24*24*44": 400, "Caja 24*24*48": 440,
            "Uhaul 24*24*20": 180, "Uhaul 24.5*24.5*27.5": 260,
            "Bici niño": 60, "Bici med": 80, "Bici adulto": 100,
            "Mochila de escuela": 40, "Mochila 5 llantas": 180, "Mochila 6 llantas": 230,
            "TV 30-39": 90, "TV 40-49": 120, "TV 50-55": 170,
            "TV 60-65": 220, "TV 70-75": 270, "TV 80-85": 320
        },
        "QRO_SLP": {
            "18gal": 100, "20gal": 110, "22gal": 130, "27gal": 150,
            "30gal": 160, "35gal": 180, "40gal": 190, "45gal": 200,
            "50gal": 220, "55gal": 240, "57gal": 250, "60gal": 260,
            "65gal": 280, "70gal": 300, "75gal": 320, "77gal": 330,
            "HD small": 100, "HD Med-Uhaul Mediana": 130, "HD Large": 160, "HD XL": 200,
            "Caja 18*18*24": 160, "Caja 22*22*22": 200, "Caja 24*24*24": 240,
            "Caja 24*24*26": 260, "Caja 24*24*44": 420, "Caja 24*24*48": 460,
            "Uhaul 24*24*20": 200, "Uhaul 24.5*24.5*27.5": 280,
            "Bici niño": 70, "Bici med": 90, "Bici adulto": 110,
            "Mochila de escuela": 50, "Mochila 5 llantas": 200, "Mochila 6 llantas": 250,
            "TV 30-39": 100, "TV 40-49": 130, "TV 50-55": 180,
            "TV 60-65": 230, "TV 70-75": 280, "TV 80-85": 330
        },
        "OTROS": {
            "18gal": 110, "20gal": 120, "22gal": 150, "27gal": 160,
            "30gal": 180, "35gal": 200, "40gal": 210, "45gal": 220,
            "50gal": 240, "55gal": 260, "57gal": 270, "60gal": 280,
            "65gal": 300, "70gal": 340, "75gal": 360, "77gal": 370,
            "HD small": 110, "HD Med-Uhaul Mediana": 150, "HD Large": 180, "HD XL": 220,
            "Caja 18*18*24": 180, "Caja 22*22*22": 220, "Caja 24*24*24": 260,
            "Caja 24*24*26": 280, "Caja 24*24*44": 460, "Caja 24*24*48": 480,
            "Uhaul 24*24*20": 220, "Uhaul 24.5*24.5*27.5": 300,
            "Bici niño": 80, "Bici med": 100, "Bici adulto": 120,
            "Mochila de escuela": 60, "Mochila 5 llantas": 220, "Mochila 6 llantas": 270,
            "TV 30-39": 120, "TV 40-49": 150, "TV 50-55": 200,
            "TV 60-65": 250, "TV 70-75": 300, "TV 80-85": 350
        }
    },
    "TX_TN": {
        "GTO": {
            "18gal": 70, "20gal": 80, "22gal": 90, "27gal": 100,
            "30gal": 110, "35gal": 130, "40gal": 140, "45gal": 150,
            "50gal": 170, "55gal": 190, "57gal": 200, "60gal": 210,
            "65gal": 230, "70gal": 250, "75gal": 270, "77gal": 280,
            "HD small": 70, "HD Med-Uhaul Mediana": 90, "HD Large": 110, "HD XL": 150,
            "Caja 18*18*24": 110, "Caja 22*22*22": 150, "Caja 24*24*24": 190,
            "Caja 24*24*26": 210, "Caja 24*24*44": 340, "Caja 24*24*48": 380,
            "Uhaul 24*24*20": 150, "Uhaul 24.5*24.5*27.5": 230,
            "Bici niño": 40, "Bici med": 60, "Bici adulto": 80,
            "Mochila de escuela": 30, "Mochila 5 llantas": 150, "Mochila 6 llantas": 200,
            "TV 30-39": 70, "TV 40-49": 90, "TV 50-55": 140,
            "TV 60-65": 190, "TV 70-75": 240, "TV 80-85": 290
        },
        "QRO_SLP": {
            "18gal": 80, "20gal": 90, "22gal": 100, "27gal": 120,
            "30gal": 130, "35gal": 150, "40gal": 160, "45gal": 170,
            "50gal": 190, "55gal": 210, "57gal": 220, "60gal": 220,
            "65gal": 250, "70gal": 270, "75gal": 290, "77gal": 300,
            "HD small": 80, "HD Med-Uhaul Mediana": 100, "HD Large": 130, "HD XL": 170,
            "Caja 18*18*24": 130, "Caja 22*22*22": 170, "Caja 24*24*24": 210,
            "Caja 24*24*26": 240, "Caja 24*24*44": 380, "Caja 24*24*48": 400,
            "Uhaul 24*24*20": 170, "Uhaul 24.5*24.5*27.5": 250,
            "Bici niño": 50, "Bici med": 70, "Bici adulto": 90,
            "Mochila de escuela": 40, "Mochila 5 llantas": 170, "Mochila 6 llantas": 220,
            "TV 30-39": 80, "TV 40-49": 100, "TV 50-55": 150,
            "TV 60-65": 200, "TV 70-75": 250, "TV 80-85": 300
        },
        "OTROS": {
            "18gal": 90, "20gal": 100, "22gal": 120, "27gal": 130,
            "30gal": 140, "35gal": 160, "40gal": 170, "45gal": 180,
            "50gal": 200, "55gal": 220, "57gal": 230, "60gal": 240,
            "65gal": 260, "70gal": 290, "75gal": 310, "77gal": 320,
            "HD small": 90, "HD Med-Uhaul Mediana": 120, "HD Large": 140, "HD XL": 180,
            "Caja 18*18*24": 140, "Caja 22*22*22": 180, "Caja 24*24*24": 220,
            "Caja 24*24*26": 240, "Caja 24*24*44": 400, "Caja 24*24*48": 420,
            "Uhaul 24*24*20": 180, "Uhaul 24.5*24.5*27.5": 260,
            "Bici niño": 60, "Bici med": 80, "Bici adulto": 100,
            "Mochila de escuela": 50, "Mochila 5 llantas": 180, "Mochila 6 llantas": 230,
            "TV 30-39": 90, "TV 40-49": 120, "TV 50-55": 170,
            "TV 60-65": 220, "TV 70-75": 270, "TV 80-85": 320
        }
    }
}

try:
    with open(os.path.join(os.path.dirname(__file__), "ciudades_mapping.json"), "r", encoding="utf-8") as f:
        mapping_data = json.load(f)
        ESTADOS = mapping_data["estados"]
        CIUDADES_POR_ESTADO = mapping_data["ciudades_por_estado"]
except Exception as e:
    print(f"Error cargando ciudades_mapping.json: {e}")
    ESTADOS = ["Guanajuato", "Querétaro", "San Luis Potosí", "Otros Estados"]
    CIUDADES_POR_ESTADO = {
        "Guanajuato": ["Dolores Hidalgo", "San Diego de la Unión", "San Luis de la Paz", "Otra Ciudad"],
        "Querétaro": ["Querétaro", "Otra Ciudad"],
        "San Luis Potosí": ["San Luis Potosí", "Otra Ciudad"],
        "Otros Estados": ["Otra Ciudad"]
    }

ORIGENES = ["Georgia", "Alabama", "Texas", "Tennessee"]
MEDIDAS_LIST = list(PRICES["GA_AL"]["GTO"].keys())

# Artículos prohibidos
PROHIBITED_KEYWORDS = [
    'arma', 'pistola', 'rifle', 'municion', 'joya', 'oro', 'plata', 'diamante', 
    'animal', 'perro', 'gato', 'serpiente', 'liquido', 'aceite', 'perfume', 
    'comida', 'perecedero', 'carne', 'fruta', 'medicamento', 'droga', 'estupefaciente',
    'dinero', 'efectivo', 'billete', 'moneda', 'explosivo', 'inflamable', 'gasolina', 
    'vape', 'cigarrillo', 'alcohol', 'pisto', 'chupe', 'veneno', 'quimico'
]

def contains_prohibited(text: str) -> bool:
    if not text: return False
    text = text.lower()
    return any(word in text for word in PROHIBITED_KEYWORDS)


def get_quote(origin_text: str, destination_text: str, ciudad_text: str, measure_key: str):
    origin_group = "GA_AL" if origin_text in ["Georgia", "Alabama"] else "TX_TN"
    # Destinos que usan tarifa GTO (Guanajuato)
    GTO_DESTS = {"Dolores Hidalgo", "San Diego de la Unión", "San Luis de la Paz"}
    if ciudad_text in GTO_DESTS:
        dest_group = "GTO"
    elif ciudad_text in ["Querétaro", "San Luis Potosí"]:
        dest_group = "QRO_SLP"
    else:
        dest_group = "OTROS"
    try:
        return PRICES[origin_group][dest_group][measure_key]
    except KeyError:
        return None


# ─── Google Sheets ─────────────────────────────────────────────────────────────
def save_to_sheets(data: dict):
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        print(f"📊 Intentando guardar en Google Sheets para: {data.get('nombre')}")
        
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_path = os.path.join(os.path.dirname(__file__), "RORS890824N37.json")
        
        if not os.path.exists(creds_path):
            print(f"❌ ERROR: Archivo de credenciales NO ENCONTRADO en: {creds_path}")
            return False
            
        creds = Credentials.from_service_account_file(creds_path, scopes=scope)
        client = gspread.authorize(creds)
        
        spreadsheet_id = "1zJ6NGUTtMGNvXoviYwKK4oyE70xnbF0K2Zv4iTqVdE8"
        print(f"📂 Abriendo hoja con ID: {spreadsheet_id}")
        
        sheet = client.open_by_key(spreadsheet_id).sheet1
        
        from datetime import datetime
        fecha_registro = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Preparar destino completo si hay ciudad específica
        destino_val = data.get("destino", "")
        ciudad = data.get("ciudad_destino", "")
        if ciudad and ciudad != "Otra Ciudad":
            destino_val = f"{destino_val} ({ciudad})"

        row = [
            data.get("nombre", ""),
            data.get("origen", ""),
            destino_val,
            data.get("medidas", ""),
            data.get("contenido", ""),
            data.get("telefono", ""),
            data.get("valor_declarado", ""),
            data.get("fragil", "No"),
            data.get("fecha_envio", ""),
            data.get("notas", ""),
            fecha_registro,
            data.get("cotizacion", ""),
            "Solicitado"
        ]
        
        print(f"📝 Insertando fila: {row}")
        sheet.append_row(row)
        print("✅ Fila insertada correctamente en Google Sheets.")
        return True
    except Exception as e:
        print(f"❌ ERROR CRÍTICO guardando en Sheets: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ─── App FastAPI ───────────────────────────────────────────────────────────────
app = FastAPI(title="Super Paqui")

# Archivos estáticos
static_path = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_path, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_path), name="static")

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat")
async def chat(request: Request):
    """Endpoint principal del chatbot — responde preguntas generales."""
    body = await request.json()
    user_message = body.get("message", "").strip()
    if not user_message:
        return JSONResponse({"response": "Por favor escribe un mensaje.", "tag": "error"})

    cb = get_chatbot()
    result = cb.chatbot_response(user_message)
    
    # Validar si la respuesta del bot o el mensaje del usuario contienen prohibidos
    if contains_prohibited(user_message):
        return JSONResponse({
            "response": "Por políticas de seguridad de la empresa, no está permitido el envío de artículos como armas, joyas, animales, dinero, líquidos o perecederos. ¿Deseas consultar sobre otro tipo de envío?",
            "tag": "prohibido",
            "confidence": 1.0
        })

    return JSONResponse({
        "response": result.get("response", "No entendí eso."),
        "tag": result.get("tag", "fallback"),
        "confidence": result.get("confidence", 0.0),
        "options": result.get("options", [])
    })


@app.get("/opciones")
async def get_opciones():
    """Devuelve las listas de opciones para el cuestionario."""
    return JSONResponse({
        "origenes": ORIGENES,
        "estados": ESTADOS,
        "ciudades_por_estado": CIUDADES_POR_ESTADO,
        "medidas": MEDIDAS_LIST
    })


@app.post("/cotizar")
async def cotizar(request: Request):
    """Calcula cotización pero NO guarda en Google Sheets todavía."""
    body = await request.json()

    origen = body.get("origen", "").strip()
    destino = body.get("destino", "").strip()
    ciudad_destino = body.get("ciudad_destino", "").strip()
    medidas = body.get("medidas", "").strip()
    contenido = body.get("contenido", "")

    # ── Validar contenido prohibido ────────────────────────────────────────────
    if contains_prohibited(contenido) or contains_prohibited(medidas):
        return JSONResponse({
            "response": "⚠️ Lo sentimos, pero no podemos transportar el contenido mencionado por políticas de seguridad. ¿Deseas cotizar un paquete con ropa, herramientas o electrónicos?",
            "is_prohibited": True
        })

    # ── Validar que el origen sea uno de los 4 estados válidos ─────────────────
    if origen not in ORIGENES:
        return JSONResponse({
            "response": (
                f"Lo sentimos, actualmente no contamos con servicio de recolección desde {origen}. 😔\n\n"
                f"Nuestros estados de origen son:\n"
                f"Georgia, Alabama, Texas y Tennessee\n\n"
                f"Para más información contáctanos directamente:\n"
                f"📞 (470) 263-6148 | WhatsApp: 418 110 7243"
            ),
            "is_invalid_origin": True
        })

    # ── Validar que el destino (estado) sea válido ─────────────────────────────
    if destino not in ESTADOS:
        return JSONResponse({
            "response": (
                f"No reconocemos el estado de destino {destino}. "
                f"Por favor selecciona uno de los estados disponibles o contáctanos al 📞 (470) 263-6148."
            ),
            "is_invalid_origin": True
        })

    # ── Validar que la ciudad de destino sea válida ──────────────────────────────
    ciudades_validas = CIUDADES_POR_ESTADO.get(destino, [])
    if ciudad_destino not in ciudades_validas:
        return JSONResponse({
            "response": (
                f"No reconocemos la ciudad de destino {ciudad_destino} para el estado {destino}. "
                f"Por favor selecciona una de las ciudades disponibles o contáctanos al 📞 (470) 263-6148."
            ),
            "is_invalid_origin": True
        })

    nombre_completo = body.get("nombre", "Cliente")
    # Extraer solo el primer nombre
    nombre = nombre_completo.split()[0] if nombre_completo else "Cliente"

    telefono = body.get("telefono", "")

    precio = get_quote(origen, destino, ciudad_destino, medidas)
    
    # Evaluar tipo de entrega
    tipo_entrega = "A Domicilio"
    if ciudad_destino and ciudad_destino != "Otra Ciudad":
        try:
            import chatbot as cb
            c_clean = ciudad_destino.strip().lower()
            # La mayoría de claves en ocurren ignoran el " (2)" si venía, pero en chatbot es limpio
            in_dom = sum(1 for k in cb.DESTINOS_DOMICILIO if c_clean in k) > 0
            in_ocu = sum(1 for k in cb.DESTINOS_OCURRE if c_clean in k) > 0
            if in_dom and in_ocu:
                tipo_entrega = "A Domicilio (o en Ocurre)"
            elif in_ocu:
                tipo_entrega = "Ocurre (Recoger en Sucursal)"
        except Exception:
            pass

    if precio:
        cotizacion_str = f"${precio}.00 USD"
        destino_final = f"{destino} ({ciudad_destino})" if ciudad_destino != "Otra Ciudad" else destino
        msg = (
            f"✅ ¡Listo, {nombre}! Tu cotización estimada es:\n\n"
            f"📦 {medidas}\n"
            f"📍 {origen} → {destino_final}\n"
            f"🚚 Entrega: {tipo_entrega}\n"
            f"💰 {cotizacion_str}\n\n"
            f"¿Deseas realizar el pedido con estos datos?"
        )
    else:
        cotizacion_str = "Consultar con asesor"
        msg = (
            f"Gracias, {nombre}. No tenemos precio automático para esa combinación. "
            f"¿Deseas que un asesor te contacte al {telefono} para una cotización personalizada? 📞"
        )

    return JSONResponse({
        "response": msg,
        "cotizacion": cotizacion_str,
        "primer_nombre": nombre
    })


@app.post("/confirmar")
async def confirmar(request: Request):
    """Guarda en Google Sheets después de la confirmación del usuario."""
    body = await request.json()
    print(f"🔔 Petición de confirmación recibida para: {body.get('nombre')}")
    
    saved = save_to_sheets(body)
    
    if saved:
        return JSONResponse({
            "response": "¡Excelente! Tu pedido ha sido registrado. Un asesor se comunicará contigo pronto. 🚚💨",
            "status": "success"
        })
    else:
        return JSONResponse({
            "response": "Tu pedido está pendiente. Por favor, comunícate directamente al 📞 (470) 263-6148 para asegurar tu lugar. Hubo un pequeño problema al registrarlo automáticamente.",
            "status": "error"
        })


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
