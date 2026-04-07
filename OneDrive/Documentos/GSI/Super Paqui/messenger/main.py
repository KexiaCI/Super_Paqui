import os
import json
import traceback
import httpx
from fastapi import FastAPI, Request, Response
from typing import Dict, Any

# Inicialización del chatbot
import chatbot as cb

# ─── Configuración FB Messenger ──────────────────────────────────────────────
VERIFY_TOKEN = os.getenv("FB_VERIFY_TOKEN", "superpaqui_token_123")
PAGE_ACCESS_TOKEN = os.getenv("FB_PAGE_ACCESS_TOKEN", "PONER_AQUI_EL_TOKEN_DE_PAGINA")

# ─── Máquina de Estados (Memoria de Sesiones) ────────────────────────────────
# sessions[sender_id] = { mode: "chat"|"quote"|"asking_name", quoteStep: 0, quoteData: {}, userName: None }
sessions = {}

# ─── Diccionarios de Precios y Lógicas (Copia de WhatsApp) ───────────────────
PRICES = {
    "GA_AL": {
        "GTO": {
            "18gal": 90, "20gal": 100, "22gal": 120, "27gal": 130, "30gal": 140, "35gal": 160, "40gal": 170, "45gal": 180,
            "50gal": 200, "55gal": 220, "57gal": 230, "60gal": 240, "65gal": 260, "70gal": 280, "75gal": 300, "77gal": 310,
            "HD small": 90, "HD Med-Uhaul Mediana": 120, "HD Large": 140, "HD XL": 180,
            "Caja 18*18*24": 140, "Caja 22*22*22": 180, "Caja 24*24*24": 220, "Caja 24*24*26": 240, "Caja 24*24*44": 400, "Caja 24*24*48": 440,
            "Uhaul 24*24*20": 180, "Uhaul 24.5*24.5*27.5": 260,
            "Bici niño": 60, "Bici med": 80, "Bici adulto": 100,
            "Mochila de escuela": 40, "Mochila 5 llantas": 180, "Mochila 6 llantas": 230,
            "TV 30-39": 90, "TV 40-49": 120, "TV 50-55": 170, "TV 60-65": 220, "TV 70-75": 270, "TV 80-85": 320
        },
        "QRO_SLP": {
            "18gal": 100, "20gal": 110, "22gal": 130, "27gal": 150, "30gal": 160, "35gal": 180, "40gal": 190, "45gal": 200,
            "50gal": 220, "55gal": 240, "57gal": 250, "60gal": 260, "65gal": 280, "70gal": 300, "75gal": 320, "77gal": 330,
            "HD small": 100, "HD Med-Uhaul Mediana": 130, "HD Large": 160, "HD XL": 200,
            "Caja 18*18*24": 160, "Caja 22*22*22": 200, "Caja 24*24*24": 240, "Caja 24*24*26": 260, "Caja 24*24*44": 420, "Caja 24*24*48": 460,
            "Uhaul 24*24*20": 200, "Uhaul 24.5*24.5*27.5": 280,
            "Bici niño": 70, "Bici med": 90, "Bici adulto": 110,
            "Mochila de escuela": 50, "Mochila 5 llantas": 200, "Mochila 6 llantas": 250,
            "TV 30-39": 100, "TV 40-49": 130, "TV 50-55": 180, "TV 60-65": 230, "TV 70-75": 280, "TV 80-85": 330
        },
        "OTROS": {
            "18gal": 110, "20gal": 120, "22gal": 150, "27gal": 160, "30gal": 180, "35gal": 200, "40gal": 210, "45gal": 220,
            "50gal": 240, "55gal": 260, "57gal": 270, "60gal": 280, "65gal": 300, "70gal": 340, "75gal": 360, "77gal": 370,
            "HD small": 110, "HD Med-Uhaul Mediana": 150, "HD Large": 180, "HD XL": 220,
            "Caja 18*18*24": 180, "Caja 22*22*22": 220, "Caja 24*24*24": 260, "Caja 24*24*26": 280, "Caja 24*24*44": 460, "Caja 24*24*48": 480,
            "Uhaul 24*24*20": 220, "Uhaul 24.5*24.5*27.5": 300,
            "Bici niño": 80, "Bici med": 100, "Bici adulto": 120,
            "Mochila de escuela": 60, "Mochila 5 llantas": 220, "Mochila 6 llantas": 270,
            "TV 30-39": 120, "TV 40-49": 150, "TV 50-55": 200, "TV 60-65": 250, "TV 70-75": 300, "TV 80-85": 350
        }
    },
    "TX_TN": {
        "GTO": {
            "18gal": 70, "20gal": 80, "22gal": 90, "27gal": 100, "30gal": 110, "35gal": 130, "40gal": 140, "45gal": 150,
            "50gal": 170, "55gal": 190, "57gal": 200, "60gal": 210, "65gal": 230, "70gal": 250, "75gal": 270, "77gal": 280,
            "HD small": 70, "HD Med-Uhaul Mediana": 90, "HD Large": 110, "HD XL": 150,
            "Caja 18*18*24": 110, "Caja 22*22*22": 150, "Caja 24*24*24": 190, "Caja 24*24*26": 210, "Caja 24*24*44": 340, "Caja 24*24*48": 380,
            "Uhaul 24*24*20": 150, "Uhaul 24.5*24.5*27.5": 230,
            "Bici niño": 40, "Bici med": 60, "Bici adulto": 80,
            "Mochila de escuela": 30, "Mochila 5 llantas": 150, "Mochila 6 llantas": 200,
            "TV 30-39": 70, "TV 40-49": 90, "TV 50-55": 140, "TV 60-65": 190, "TV 70-75": 240, "TV 80-85": 290
        },
        "QRO_SLP": {
            "18gal": 80, "20gal": 90, "22gal": 100, "27gal": 120, "30gal": 130, "35gal": 150, "40gal": 160, "45gal": 170,
            "50gal": 190, "55gal": 210, "57gal": 220, "60gal": 220, "65gal": 250, "70gal": 270, "75gal": 290, "77gal": 300,
            "HD small": 80, "HD Med-Uhaul Mediana": 100, "HD Large": 130, "HD XL": 170,
            "Caja 18*18*24": 130, "Caja 22*22*22": 170, "Caja 24*24*24": 210, "Caja 24*24*26": 240, "Caja 24*24*44": 380, "Caja 24*24*48": 400,
            "Uhaul 24*24*20": 170, "Uhaul 24.5*24.5*27.5": 250,
            "Bici niño": 50, "Bici med": 70, "Bici adulto": 90,
            "Mochila de escuela": 40, "Mochila 5 llantas": 170, "Mochila 6 llantas": 220,
            "TV 30-39": 80, "TV 40-49": 100, "TV 50-55": 150, "TV 60-65": 200, "TV 70-75": 250, "TV 80-85": 300
        },
        "OTROS": {
            "18gal": 90, "20gal": 100, "22gal": 120, "27gal": 130, "30gal": 140, "35gal": 160, "40gal": 170, "45gal": 180,
            "50gal": 200, "55gal": 220, "57gal": 230, "60gal": 240, "65gal": 260, "70gal": 290, "75gal": 310, "77gal": 320,
            "HD small": 90, "HD Med-Uhaul Mediana": 120, "HD Large": 140, "HD XL": 180,
            "Caja 18*18*24": 140, "Caja 22*22*22": 180, "Caja 24*24*24": 220, "Caja 24*24*26": 240, "Caja 24*24*44": 400, "Caja 24*24*48": 420,
            "Uhaul 24*24*20": 180, "Uhaul 24.5*24.5*27.5": 260,
            "Bici niño": 60, "Bici med": 80, "Bici adulto": 100,
            "Mochila de escuela": 50, "Mochila 5 llantas": 180, "Mochila 6 llantas": 230,
            "TV 30-39": 90, "TV 40-49": 120, "TV 50-55": 170, "TV 60-65": 220, "TV 70-75": 270, "TV 80-85": 320
        }
    }
}

try:
    with open(os.path.join(os.path.dirname(__file__), "ciudades_mapping.json"), "r", encoding="utf-8") as f:
        mapping_data = json.load(f)
        ESTADOS = mapping_data["estados"]
        CIUDADES_POR_ESTADO = mapping_data["ciudades_por_estado"]
except Exception as e:
    ESTADOS = ["Guanajuato", "Querétaro", "San Luis Potosí", "Otros Estados"]
    CIUDADES_POR_ESTADO = {
        "Guanajuato": ["Dolores Hidalgo", "San Diego de la Unión", "San Luis de la Paz", "Otra Ciudad"],
        "Querétaro": ["Querétaro", "Otra Ciudad"],
        "San Luis Potosí": ["San Luis Potosí", "Otra Ciudad"],
        "Otros Estados": ["Otra Ciudad"]
    }

ORIGENES = ["Georgia", "Alabama", "Texas", "Tennessee"]
MEDIDAS_LIST = list(PRICES["GA_AL"]["GTO"].keys())

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

def save_to_sheets(data: dict):
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_path = os.path.join(os.path.dirname(__file__), "RORS890824N37.json")
        if not os.path.exists(creds_path):
            return False
            
        creds = Credentials.from_service_account_file(creds_path, scopes=scope)
        client = gspread.authorize(creds)
        spreadsheet_id = "1zJ6NGUTtMGNvXoviYwKK4oyE70xnbF0K2Zv4iTqVdE8"
        sheet = client.open_by_key(spreadsheet_id).sheet1
        from datetime import datetime
        fecha_registro = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
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
            "No", # Fragil por defecto
            "",   # fecha envio default envia empty
            data.get("notas", ""),
            fecha_registro,
            data.get("cotizacion", ""),
            "Solicitado (Messenger)"
        ]
        sheet.append_row(row)
        return True
    except Exception as e:
        traceback.print_exc()
        return False

# ─── FastAPI Webhook ─────────────────────────────────────────────────────────

app = FastAPI(title="Super Paqui Messenger Bot")

async def send_fb_message(sender_id: str, text: str, quick_replies: list = None):
    url = f"https://graph.facebook.com/v18.0/me/messages?access_token={PAGE_ACCESS_TOKEN}"
    payload = {
        "recipient": {"id": sender_id},
        "message": {"text": text}
    }
    
    if quick_replies:
        qr_list = []
        for qr in quick_replies:
            qr_list.append({
                "content_type": "text",
                "title": qr[:20], # FB title limit is 20 chars
                "payload": qr
            })
        payload["message"]["quick_replies"] = qr_list

    async with httpx.AsyncClient() as client:
        res = await client.post(url, json=payload)
        print(f"Enviado {res.status_code}: {res.text}")

def get_session(sender_id: str):
    if sender_id not in sessions:
        # Initial greeting mode
        sessions[sender_id] = {
            "mode": "asking_name",
            "quoteStep": 0,
            "quoteData": {},
            "userName": None
        }
    return sessions[sender_id]

def get_current_quote_steps(quote_data):
    steps = [
        {
            "key": "telefono",
            "ask": "¡Perfecto! Vamos a cotizar tu envío. 😊\n\n¿Cuál es tu número de teléfono para contactarte?",
            "options": []
        },
        {
            "key": "origen",
            "ask": "¿Desde qué estado de EE.UU. envías?",
            "options": ["Tennessee", "Texas", "Alabama", "Georgia"]
        },
        {
            "key": "destino_rapido",
            "ask": "¿A qué destino de México va el paquete?",
            "options": ["Dolores", "San Diego de la Unión", "San Luis de la Paz", "Querétaro", "San Luis Potosí", "Otros"]
        }
    ]
    
    if quote_data.get("destino_rapido") == "Otros":
        estados_texto_list = "\n".join(f"• {e}" for e in ESTADOS)
        steps.append({
            "key": "destino",
            "ask": f"Escribe el estado de México al que envías. Tenemos cobertura en:\n\n{estados_texto_list}",
            "options": []
        })
        
        steps.append({
            "key": "ciudad_destino",
            "ask": "Escribe a qué ciudad de ese estado se enviará:",
            "options": []
        })
        
    steps.append({
        "key": "medidas",
        "ask": "¿Qué tipo de paquete o medida es? Escribe el tipo o galones aproximados (ej. 'Caja 24x24x24' o '20gal')",
        "options": []
    })
    
    steps.append({
        "key": "contenido",
        "ask": "¿Qué contenido va en el paquete? (ropa, electrónicos, etc.)",
        "options": []
    })
    
    return steps

async def process_quote_step(sender_id: str, state: dict, user_text: str = None):
    q_data = state["quoteData"]
    current_steps = get_current_quote_steps(q_data)
    step_idx = state["quoteStep"]
    
    # Evaluar si el usuario acaba de responder un paso
    if user_text is not None and step_idx < len(current_steps):
        curr_step = current_steps[step_idx]
        
        if contains_prohibited(user_text):
            await send_fb_message(sender_id, "⚠️ Lo sentimos, pero no podemos transportar ese tipo de artículos por políticas de seguridad de la empresa.\n\n¿Deseas enviar otro tipo de contenido para continuar tu cotización?", quick_replies=["Sí, intentar de nuevo", "No, cancelar"])
            state["mode"] = "awaiting_retry"
            return
        
        q_data[curr_step["key"]] = user_text
        
        # Special mapping for destino_rapido if it's not "Otros"
        if curr_step["key"] == "destino_rapido" and user_text != "Otros":
            if user_text == "Dolores":
                q_data["destino"] = "Guanajuato"
                q_data["ciudad_destino"] = "Dolores Hidalgo"
            elif user_text == "San Diego de la Unión":
                q_data["destino"] = "Guanajuato"
                q_data["ciudad_destino"] = "San Diego de la Unión"
            elif user_text == "San Luis de la Paz":
                q_data["destino"] = "Guanajuato"
                q_data["ciudad_destino"] = "San Luis de la Paz"
            elif user_text in ["Querétaro", "San Luis Potosí"]:
                q_data["destino"] = user_text
                q_data["ciudad_destino"] = user_text
                
        state["quoteStep"] += 1
        step_idx += 1
        
        # Actualizamos pasos en caso de que "Otros" se haya activado
        current_steps = get_current_quote_steps(q_data)

    # Lanzar la siguiente pregunta o finalizar
    if step_idx < len(current_steps):
        next_step = current_steps[step_idx]
        options = next_step["options"]
        ask_text = next_step["ask"]
        
        if next_step["key"] == "ciudad_destino":
             target_state = q_data.get("destino", "")
             cities = []
             for e in CIUDADES_POR_ESTADO.keys():
                 import unicodedata
                 def norm(s): return unicodedata.normalize('NFD', s).encode('ascii', 'ignore').decode('utf-8').lower()
                 if norm(e) == norm(target_state) or norm(target_state) in norm(e):
                     cities = CIUDADES_POR_ESTADO[e]
                     break
             if not cities:
                 cities = ["Otra Ciudad"]
                 
             cities_str = "\n".join(f"• {c}" for c in cities)
             ask_text = f"En ese estado tenemos cobertura en:\n\n{cities_str}\n\nPor favor, escribe el nombre de la ciudad."
             options = []
             
        await send_fb_message(sender_id, ask_text, quick_replies=options if options else None)
    else:
         # Calcular cotización
        await calculate_and_finish_quote(sender_id, state)

async def calculate_and_finish_quote(sender_id: str, state: dict):
    q_data = state["quoteData"]
    precio = get_quote(q_data.get("origen", ""), q_data.get("destino", ""), q_data.get("ciudad_destino", ""), q_data.get("medidas", ""))
    
    primer_nombre = state["userName"]
    destino_final = q_data.get("destino")
    if q_data.get("ciudad_destino") != "Otra Ciudad":
        destino_final += f" ({q_data.get('ciudad_destino')})"
        
    if precio:
        # Evaluar tipo de entrega
        tipo_entrega = "A Domicilio"
        if q_data.get("ciudad_destino") and q_data.get("ciudad_destino") != "Otra Ciudad":
            try:
                import chatbot as cb
                c_clean = q_data.get("ciudad_destino").strip().lower()
                in_dom = sum(1 for k in cb.DESTINOS_DOMICILIO if c_clean in k) > 0
                in_ocu = sum(1 for k in cb.DESTINOS_OCURRE if c_clean in k) > 0
                if in_dom and in_ocu:
                    tipo_entrega = "A Domicilio (o en Ocurre)"
                elif in_ocu:
                    tipo_entrega = "Ocurre (Recoger en Sucursal)"
            except Exception:
                pass
                
        cotizacion_str = f"${precio}.00 USD"
        q_data["cotizacion"] = cotizacion_str
        msg = (
            f"✅ ¡Listo, {primer_nombre}! Tu cotización estimada es:\n\n"
            f"📦 {q_data.get('medidas')}\n"
            f"📍 {q_data.get('origen')} → {destino_final}\n"
            f"🚚 Entrega: {tipo_entrega}\n"
            f"💰 {cotizacion_str}\n\n"
            f"¿Deseas realizar el pedido con estos datos?"
        )
        await send_fb_message(sender_id, msg, quick_replies=["✅ Sí, hacer pedido", "❌ No, gracias"])
        state["mode"] = "awaiting_confirmation"
    else:
        msg = (
            f"Gracias, {primer_nombre}. No tenemos precio automático para esa combinación exacta. "
            f"¿Deseas que un asesor te contacte al {q_data.get('telefono')} para una cotización personalizada? 📞"
        )
        await send_fb_message(sender_id, msg, quick_replies=["Sí, que me contacten", "No, gracias"])
        state["mode"] = "awaiting_confirmation"

async def process_chat_message(sender_id: str, text: str, state: dict):
    # Procesamiento inicial de nombre
    if state["mode"] == "asking_name":
        import re
        name = text.strip()
        name = re.sub(r'^(hola|buenas tardes|buenos d[ií]as|buen d[ií]a|qué tal|saludos)[,.\s]+', '', name, flags=re.IGNORECASE)
        name = re.split(r'(?:yo\s+)?soy\s+|me\s+llamo\s+|mi\s+nombre\s+es\s+', name, flags=re.IGNORECASE)[-1]
        name = name.strip()
        if name:
            name = name.capitalize()
        else:
            name = "Amigo(a)"
        state["userName"] = name
        state["quoteData"]["nombre"] = state["userName"]
        state["mode"] = "chat"
        await send_fb_message(sender_id, f"¡Mucho gusto, {state['userName']}! Puedo ayudarte con la cotización de tus envíos y resolver tus dudas.\n\n¿En qué te puedo ayudar hoy?", quick_replies=["📦 Realizar pedido", "📞 Contacto"])
        return

    # Check cotización trigger manual
    tl = text.lower()
    if "cotizar" in tl or "pedido" in tl:
        state["mode"] = "quote"
        state["quoteStep"] = 0
        state["quoteData"] = {"nombre": state.get("userName", "Cliente")}
        await process_quote_step(sender_id, state)
        return

    # Check de prohibidos manual en el chat libre
    if contains_prohibited(text):
        await send_fb_message(sender_id, "Por políticas de seguridad de la empresa, no está permitido el envío de artículos peligrosos o prohibidos. ¿Deseas consultar sobre otro tipo de envío?", quick_replies=["📦 Realizar pedido", "📞 Contacto"])
        return

    # Lógica de chatbot AI
    # Agregando el nombre del usuario al chatbot prompt si fuera necesario
    result = cb.chatbot_response(text)
    response_msg = result.get("response", "No entendí eso.")
    
    if result.get("options") and len(result["options"]) > 0:
         options = result["options"] if len(result["options"]) <= 13 else result["options"][:13]
         await send_fb_message(sender_id, response_msg, quick_replies=options)
    else:
        await send_fb_message(sender_id, response_msg, quick_replies=["📦 Realizar pedido", "📞 Contacto"])

@app.get("/")
def read_root():
    return {"status": "ok", "service": "Super Paqui Messenger Bot"}

@app.get("/webhook")
def verify_webhook(request: Request):
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode and token:
        if mode == "subscribe" and token == VERIFY_TOKEN:
            print("WEBHOOK_VERIFIED")
            return Response(content=challenge, status_code=200)
        else:
            return Response(status_code=403)
    return Response(status_code=400)

@app.post("/webhook")
async def receive_webhook(request: Request):
    body = await request.json()

    if body.get("object") == "page":
        for entry in body.get("entry", []):
            for event in entry.get("messaging", []):
                sender_id = event["sender"]["id"]
                
                # Check for message text or quick reply postback
                text = None
                if "message" in event and "text" in event["message"]:
                     text = event["message"]["text"]
                     
                     if "quick_reply" in event["message"]:
                         text = event["message"]["quick_reply"]["payload"]

                if text:
                    state = get_session(sender_id)
                    
                    if state["mode"] == "quote":
                        await process_quote_step(sender_id, state, text)
                    elif state["mode"] == "awaiting_retry":
                        if "Sí" in text:
                            state["mode"] = "quote"
                            # Re-preguntar el actual paso de la cotización
                            await process_quote_step(sender_id, state, None)
                        else:
                            await send_fb_message(sender_id, "¡Entendido! La cotización ha sido cancelada.", quick_replies=["📦 Realizar pedido", "📞 Contacto"])
                            state["mode"] = "chat"
                    elif state["mode"] == "awaiting_confirmation":
                        if "Sí" in text:
                            # Confirm order to Google Sheets
                            await send_fb_message(sender_id, "Registrando pedido...")
                            saved = save_to_sheets(state["quoteData"])
                            if saved:
                                await send_fb_message(sender_id, "¡Excelente! Tu pedido ha sido registrado. Un asesor se comunicará contigo pronto. 🚚💨", quick_replies=["📞 Contacto", "Nueva cotización"])
                            else:
                                await send_fb_message(sender_id, "Tu pedido está pendiente. Por favor, comunícate directamente al 📞 (470) 263-6148. Hubo un pequeño error al guardarlo.", quick_replies=["📞 Contacto"])
                            state["mode"] = "chat"
                        else:
                            await send_fb_message(sender_id, "¡Entendido! Si cambias de opinión aquí estaré.", quick_replies=["📦 Realizar pedido", "📞 Contacto"])
                            state["mode"] = "chat"
                    else:
                        await process_chat_message(sender_id, text, state)
                        
        return Response(content="EVENT_RECEIVED", status_code=200)
    else:
        return Response(status_code=404)
