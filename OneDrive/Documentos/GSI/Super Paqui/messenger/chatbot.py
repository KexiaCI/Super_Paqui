import json
import random
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

"""
Superpacky Chatbot (Transformer, robusto y "no inventa")
-------------------------------------------------------
Este chatbot NO genera texto libre con un LLM. En su lugar:
1) Detecta reglas de seguridad (prohibidos) y preguntas de identidad por regex.
2) Clasifica intents con un *Transformer encoder* (SentenceTransformer) + clasificador supervisado (LogReg).
3) Responde SOLO con respuestas predefinidas del intents.json (o mensajes fijos de seguridad/fallback).

Ventajas vs. KNN por similitud:
- Menos confusiones entre intents porque aprende un separador supervisado.
- Mejor calibración de confianza.
- Sigue siendo rápido y ligero para producción.

Requisitos:
pip install sentence-transformers scikit-learn numpy
"""

# ======================
# CONFIGURACIÓN
# ======================

MODEL_NAME = os.getenv("SUPERPACKY_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Umbrales de confianza (puedes ajustarlos)
MIN_PROBA_TO_ACCEPT = float(os.getenv("SUPERPACKY_MIN_PROBA", "0.45"))
MARGIN_TO_ACCEPT = float(os.getenv("SUPERPACKY_MARGIN", "0.10"))  # diferencia entre top1 y top2

# Si un intent está muy "sensible" a confusión, puedes exigir más confianza:
PER_TAG_MIN_PROBA = {
    # Ejemplos (ajusta según tu data):
    # "precios": 0.62,
    # "sucursales": 0.62,
}

# ======================
# CARGA DE INTENTS
# ======================

_intents_path = os.path.join(os.path.dirname(__file__), "intents.json")
with open(_intents_path, "r", encoding="utf-8") as f:
    intents = json.load(f)

# Mapas
responses_map: Dict[str, List[str]] = {i["tag"]: i.get("responses", []) for i in intents}
keywords_map: Dict[str, List[str]] = {i["tag"]: i.get("keywords", []) for i in intents}

fallback_responses = responses_map.get("fallback", [
    "Lo siento, no entendí tu consulta. ¿Quieres una cotización, horarios o información de envío?"
])

# ======================
# REGLAS / SEGURIDAD
# ======================

PROHIBITED_ITEMS = [
    'arma', 'pistola', 'rifle', 'municion', 'joya', 'oro', 'plata', 'diamante', 'joyeria', 'joyas', 'anillo', 'arete', 'collar',
    'animal', 'perro', 'gato', 'serpiente', 'liquido', 'aceite', 'perfume', 'perrito', 'mascota', 'pajaro', 'pajarito', 'tortuga',
    'comida', 'perecedero', 'carne', 'fruta', 'medicamento', 'droga', 'estupefaciente', 'medicinas', 'pastillas', 'jarabe',
    'dinero', 'efectivo', 'billete', 'moneda', 'explosivo', 'inflamable', 'gasolina', 'gas', 'gas LP', 'gas natural', 'gas del tanque', 'gas butano', 'gas propano',
    'vape', 'cigarrillo', 'alcohol', 'pisto', 'chupe', 'veneno', 'quimico', 'quimicos', 'quimico industrial'
]

PERMITTED_ITEMS = [
    'ropa', 'zapatos', 'tenis', 'accesorios', 'maquillaje', 'skincare', 'cremas', 'cosmeticos', 'juguetes',
    'herramientas', 'electronicos', 'electrodomesticos', 'licuadora', 'plancha', 'pantalla', 'tv',
    'llaves', 'documentos', 'papeleria', 'articulos de hogar', 'libros', 'mochilas'
]

def _contains_prohibited(text: str) -> Optional[str]:
    tl = (text or "").lower()
    for item in PROHIBITED_ITEMS:
        if item in tl:
            return item
    return None

# Saludos — variantes coloquiales y formales
SALUDO_RE = re.compile(
    r"\b(hola[a]*|buenas|buenos\s+d[ií]as|buenas\s+tardes|buenas\s+noches|qu[eé]\s+tal|hey|ey|hi|hello|qu[eé]\s+onda|qu[eé]\s+hay|buen\s+d[ií]a|saludos)\b",
    re.IGNORECASE
)

# Identidad / "quién eres" — acepta ¿?, tildes y variantes comunes
IDENTITY_RE = re.compile(
    r"(¿|\b)("
    r"c[oó]mo\s+te\s+llamas|"
    r"cu[aá]l\s+es\s+tu\s+nombre|"
    r"qui[eé]n\s+eres|"
    r"qu[eé]\s+eres|"
    r"eres\s+un\s+bot|"
    r"eres\s+(?:una?\s+)?[iI][aA]|"
    r"eres\s+(?:un\s+)?chatbot|"
    r"tu\s+nombre|"
    r"c[oó]mo\s+te\s+llaman|"
    r"cu[aá]l\s+es\s+tu\s+identidad|"
    r"con\s+qui[eé]n\s+(estoy\s+)?(hablo|hablando)|eres\s+(?:una?\s+)?inteligencia\s+artificial"
    r")(?:\b|\?|$)",
    re.IGNORECASE
)

IDENTITY_RESPONSES = [
    "Soy Superpacky, el asistente de paquetería San Diego de la Unión y estoy para servirte. 🚚✨",
    "¡Hola! Soy Superpacky, el asistente de Paquetería San Diego de la Unión. ¿En qué te apoyo? 😊",
    "Soy Superpacky 🤖🚚, tu asistente de envíos de Paquetería San Diego de la Unión. Estoy para ayudarte.",
    "Soy Superpacky, tu asistente virtual de Paquetería San Diego de la Unión. ¿Quieres cotizar un envío?",
    "¡A tus órdenes! Soy Superpacky, el asistente de Paquetería San Diego de la Unión.",
    "Soy Superpacky, un asistente de atención y cotizaciones de Paquetería San Diego de la Unión.",
    "Me llamo Superpacky. Soy el asistente de Paquetería San Diego de la Unión y estoy para servirte.",
    "Soy Superpacky, un chatbot de Paquetería San Diego de la Unión. Puedo orientarte y ayudarte a cotizar.",
    "Aquí Superpacky 😄, asistente de Paquetería San Diego de la Unión. Dime qué necesitas.",
    "Soy Superpacky, tu asistente de envíos. Si me dices origen, destino y medidas, te ayudo a cotizar."
]

# (Opcional) si quieres “anclar” intent de precios/cotizar para que pase por /cotizar:
QUOTE_HINT_RE = re.compile(r"\b(cotiza|cotizar|costo|cu[aá]nto|precio|tarifa)\b", re.IGNORECASE)
BRANCH_HINT_RE = re.compile(r"\b(sucursal|sucursales|oficina|oficinas|ubicaci[oó]n|direcci[oó]n|puntos?)\b", re.IGNORECASE)

import re

# ======================
# CATÁLOGO DE TAMAÑOS (fuente: tus etiquetas)
# ======================

# Regex para capturar "Caja 24*24*24" y variantes
BOX_DIM_RE = re.compile(
    r"\b(caja|uhaul)\s*(\d+(?:\.\d+)?)\s*[*xX]\s*(\d+(?:\.\d+)?)\s*[*xX]\s*(\d+(?:\.\d+)?)\b",
    re.IGNORECASE
)

# Regex para "TV 50-55"
TV_RE = re.compile(r"\btv\s*(\d+)\s*[-–]\s*(\d+)\b", re.IGNORECASE)

# Regex para "18gal", "20 gal", "22g"
GAL_RE = re.compile(r"\b(\d{2})\s*(gal|gals|galones)\b", re.IGNORECASE)

# Etiquetas manejadas en tu sistema (lo que sí sabemos con certeza)
SIZE_KB = {
    # Gallon bins: NO inventamos dimensiones, solo categoría/capacidad
    "18gal": {
        "tipo": "Contenedor por capacidad",
        "unidad": "galones",
        "detalle": "Categoría de contenedor de 18 galones (capacidad). Las medidas físicas varían por marca/modelo.",
        "para_cotizar": "Usa la categoría: 18gal"
    },
    "20gal": {"tipo":"Contenedor por capacidad","unidad":"galones","detalle":"Categoría de 20 galones (capacidad). Medidas físicas varían.","para_cotizar":"Usa la categoría: 20gal"},
    "22gal": {"tipo":"Contenedor por capacidad","unidad":"galones","detalle":"Categoría de 22 galones (capacidad). Medidas físicas varían.","para_cotizar":"Usa la categoría: 22gal"},
    "27gal": {"tipo":"Contenedor por capacidad","unidad":"galones","detalle":"Categoría de 27 galones (capacidad). Medidas físicas varían.","para_cotizar":"Usa la categoría: 27gal"},
    "30gal": {"tipo":"Contenedor por capacidad","unidad":"galones","detalle":"Categoría de 30 galones (capacidad). Medidas físicas varían.","para_cotizar":"Usa la categoría: 30gal"},
    "35gal": {"tipo":"Contenedor por capacidad","unidad":"galones","detalle":"Categoría de 35 galones (capacidad). Medidas físicas varían.","para_cotizar":"Usa la categoría: 35gal"},
    "40gal": {"tipo":"Contenedor por capacidad","unidad":"galones","detalle":"Categoría de 40 galones (capacidad). Medidas físicas varían.","para_cotizar":"Usa la categoría: 40gal"},
    "45gal": {"tipo":"Contenedor por capacidad","unidad":"galones","detalle":"Categoría de 45 galones (capacidad). Medidas físicas varían.","para_cotizar":"Usa la categoría: 45gal"},
    "50gal": {"tipo":"Contenedor por capacidad","unidad":"galones","detalle":"Categoría de 50 galones (capacidad). Medidas físicas varían.","para_cotizar":"Usa la categoría: 50gal"},
    "55gal": {"tipo":"Contenedor por capacidad","unidad":"galones","detalle":"Categoría de 55 galones (capacidad). Medidas físicas varían.","para_cotizar":"Usa la categoría: 55gal"},
    "57gal": {"tipo":"Contenedor por capacidad","unidad":"galones","detalle":"Categoría de 57 galones (capacidad). Medidas físicas varían.","para_cotizar":"Usa la categoría: 57gal"},
    "60gal": {"tipo":"Contenedor por capacidad","unidad":"galones","detalle":"Categoría de 60 galones (capacidad). Medidas físicas varían.","para_cotizar":"Usa la categoría: 60gal"},
    "65gal": {"tipo":"Contenedor por capacidad","unidad":"galones","detalle":"Categoría de 65 galones (capacidad). Medidas físicas varían.","para_cotizar":"Usa la categoría: 65gal"},
    "70gal": {"tipo":"Contenedor por capacidad","unidad":"galones","detalle":"Categoría de 70 galones (capacidad). Medidas físicas varían.","para_cotizar":"Usa la categoría: 70gal"},
    "75gal": {"tipo":"Contenedor por capacidad","unidad":"galones","detalle":"Categoría de 75 galones (capacidad). Medidas físicas varían.","para_cotizar":"Usa la categoría: 75gal"},
    "77gal": {"tipo":"Contenedor por capacidad","unidad":"galones","detalle":"Categoría de 77 galones (capacidad). Medidas físicas varían.","para_cotizar":"Usa la categoría: 77gal"},

    # Cajas: aquí SÍ hay medidas explícitas en la etiqueta (no se inventa nada)
    "Caja 18*18*24": {"tipo":"Caja con medidas","unidad":"pulgadas (según etiqueta)","detalle":"Medidas: 18 × 18 × 24"},
    "Caja 22*22*22": {"tipo":"Caja con medidas","unidad":"pulgadas (según etiqueta)","detalle":"Medidas: 22 × 22 × 22"},
    "Caja 24*24*24": {"tipo":"Caja con medidas","unidad":"pulgadas (según etiqueta)","detalle":"Medidas: 24 × 24 × 24"},
    "Caja 24*24*26": {"tipo":"Caja con medidas","unidad":"pulgadas (según etiqueta)","detalle":"Medidas: 24 × 24 × 26"},
    "Caja 24*24*44": {"tipo":"Caja con medidas","unidad":"pulgadas (según etiqueta)","detalle":"Medidas: 24 × 24 × 44"},
    "Caja 24*24*48": {"tipo":"Caja con medidas","unidad":"pulgadas (según etiqueta)","detalle":"Medidas: 24 × 24 × 48"},
    "Uhaul 24*24*20": {"tipo":"Caja con medidas (Uhaul)","unidad":"pulgadas (según etiqueta)","detalle":"Medidas: 24 × 24 × 20"},
    "Uhaul 24.5*24.5*27.5": {"tipo":"Caja con medidas (Uhaul)","unidad":"pulgadas (según etiqueta)","detalle":"Medidas: 24.5 × 24.5 × 27.5"},

    # TVs: rango de pulgadas (tamaño de pantalla)
    "TV 30-39": {"tipo":"Televisión por tamaño de pantalla","unidad":"pulgadas","detalle":"Rango de pantalla: 30 a 39 pulgadas"},
    "TV 40-49": {"tipo":"Televisión por tamaño de pantalla","unidad":"pulgadas","detalle":"Rango de pantalla: 40 a 49 pulgadas"},
    "TV 50-55": {"tipo":"Televisión por tamaño de pantalla","unidad":"pulgadas","detalle":"Rango de pantalla: 50 a 55 pulgadas"},
    "TV 60-65": {"tipo":"Televisión por tamaño de pantalla","unidad":"pulgadas","detalle":"Rango de pantalla: 60 a 65 pulgadas"},
    "TV 70-75": {"tipo":"Televisión por tamaño de pantalla","unidad":"pulgadas","detalle":"Rango de pantalla: 70 a 75 pulgadas"},
    "TV 80-85": {"tipo":"Televisión por tamaño de pantalla","unidad":"pulgadas","detalle":"Rango de pantalla: 80 a 85 pulgadas"},

    # Bicis / mochilas / HD: NO hay medidas explícitas => respuesta guiada
    "Bici niño": {"tipo":"Bicicleta (categoría)","detalle":"Categoría: bicicleta de niño. Para medidas exactas, indica rodada o largo/ancho/alto."},
    "Bici med": {"tipo":"Bicicleta (categoría)","detalle":"Categoría: bicicleta mediana. Para medidas exactas, indica rodada o largo/ancho/alto."},
    "Bici adulto": {"tipo":"Bicicleta (categoría)","detalle":"Categoría: bicicleta de adulto. Para medidas exactas, indica rodada o largo/ancho/alto."},
    "Mochila de escuela": {"tipo":"Mochila (categoría)","detalle":"Categoría: mochila escolar. Si quieres validar tamaño, dime medidas aproximadas."},
    "Mochila 5 llantas": {"tipo":"Mochila (categoría)","detalle":"Categoría: mochila con 5 llantas. Si quieres validar tamaño, dime medidas aproximadas."},
    "Mochila 6 llantas": {"tipo":"Mochila (categoría)","detalle":"Categoría: mochila con 6 llantas. Si quieres validar tamaño, dime medidas aproximadas."},
    "HD small": {"tipo":"Caja/paquete tipo HD","detalle":"Categoría: HD small. Si necesitas medidas exactas, dime largo/ancho/alto o el modelo de caja."},
    "HD Med-Uhaul Mediana": {"tipo":"Caja/paquete tipo HD","detalle":"Categoría: HD Med / Uhaul mediana. Si necesitas medidas exactas, dime el modelo o medidas."},
    "HD Large": {"tipo":"Caja/paquete tipo HD","detalle":"Categoría: HD Large. Si necesitas medidas exactas, dime largo/ancho/alto o el modelo."},
    "HD XL": {"tipo":"Caja/paquete tipo HD","detalle":"Categoría: HD XL. Si necesitas medidas exactas, dime largo/ancho/alto o el modelo."},
}

import re
from typing import Optional, Tuple, List, Dict

# ------------- PARSEO DE MEDIDAS -------------
DIM_UNIT_RE = re.compile(r"\b(cm|cent[ií]metros?|mm|mil[ií]metros?|in|inch|pulgadas?)\b", re.IGNORECASE)

# Caso: "12x18x20", "12 * 18 * 20", "12 X 18 X 20"
DIM_TRIPLE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[*xX]\s*(\d+(?:\.\d+)?)\s*[*xX]\s*(\d+(?:\.\d+)?)")

# Caso: "largo 12, ancho 18, alto 20" (con variaciones)
DIM_LWH_RE = re.compile(
    r"(largo|longitud)\s*[:=]?\s*(\d+(?:\.\d+)?)\D+"
    r"(ancho)\s*[:=]?\s*(\d+(?:\.\d+)?)\D+"
    r"(alto|altura)\s*[:=]?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE
)

def parse_user_dimensions(text: str) -> Tuple[Optional[Tuple[float, float, float]], Optional[str]]:
    """
    Detecta medidas aunque el usuario escriba:
    - "el largo son 12, ancho 18 y alto 20"
    - "largo: 12 ancho: 18 alto: 20"
    - "12x18x20"
    - en cualquier orden: "alto 20 largo 12 ancho 18"
    """
    t = (text or "").lower()

    # unidad
    unit = None
    mu = DIM_UNIT_RE.search(t)
    if mu:
        u = mu.group(1).lower()
        if u.startswith("cm") or "cent" in u:
            unit = "cm"
        elif u.startswith("mm") or "mil" in u:
            unit = "mm"
        elif u in ("in", "inch") or "pulg" in u:
            unit = "in"

    # 1) triple 12x18x20
    m = DIM_TRIPLE_RE.search(t)
    if m:
        a, b, c = float(m.group(1)), float(m.group(2)), float(m.group(3))
        return (a, b, c), unit

    # 2) etiquetas sueltas en cualquier orden, tolerante a "son/es/mide/=" etc.
    # ejemplo: "largo son 12", "ancho: 18", "alto 20"
    tag_re = re.compile(r"\b(largo|longitud|ancho|alto|altura)\b[^0-9]{0,15}(\d+(?:\.\d+)?)")
    found = tag_re.findall(t)

    vals = {}
    for k, v in found:
        k = k.strip()
        vals[k] = float(v)

    # normaliza claves
    L = vals.get("largo") or vals.get("longitud")
    W = vals.get("ancho")
    H = vals.get("alto") or vals.get("altura")

    if L is not None and W is not None and H is not None:
        return (L, W, H), unit

    return None, unit

# ------------- CATÁLOGO: EXTRAER CAJAS REALES DESDE TUS LLAVES -------------
BOX_KEY_RE = re.compile(r"^(caja|uhaul)\s*(\d+(?:\.\d+)?)\*(\d+(?:\.\d+)?)\*(\d+(?:\.\d+)?)$", re.IGNORECASE)

def build_box_catalog_from_measure_keys(measure_keys: List[str]) -> List[Dict]:
    """
    De una lista de medidas (strings), extrae solo cajas tipo "Caja A*B*C" y "Uhaul A*B*C".
    """
    catalog = []
    for k in measure_keys:
        s = (k or "").strip()
        m = BOX_KEY_RE.match(s.replace("x", "*").replace("X","*"))
        if not m:
            continue
        kind = m.group(1).title()  # Caja / Uhaul
        a,b,c = float(m.group(2)), float(m.group(3)), float(m.group(4))
        dims = sorted([a,b,c])  # ordenamos para comparar sin importar orientación
        catalog.append({"key": s, "kind": kind, "dims_sorted": dims})
    # ordena por volumen (aprox) ascendente
    catalog.sort(key=lambda x: x["dims_sorted"][0]*x["dims_sorted"][1]*x["dims_sorted"][2])
    return catalog


def suggest_box(measure_keys: List[str], user_dims: Tuple[float,float,float], unit: Optional[str]) -> Dict:
    """
    Sugiere la caja más pequeña que "quepa", comparando contra cajas del catálogo.
    NO asume unidad si no viene: si unit is None, devuelve sugerencia doble.
    """
    catalog = build_box_catalog_from_measure_keys(measure_keys)
    if not catalog:
        return {"ok": False, "msg": "No encontré cajas con formato Caja A*B*C en tu catálogo."}

    def _pick(dims_sorted):
        # busca la primera caja que cubra las 3 dimensiones
        for item in catalog:
            bd = item["dims_sorted"]
            if bd[0] >= dims_sorted[0] and bd[1] >= dims_sorted[1] and bd[2] >= dims_sorted[2]:
                return item
        return None

    ud = sorted(list(user_dims))

    # si la unidad es mm, conviértelo a cm para orientar (solo para el mensaje); pero NO lo casamos a tu catálogo
    # (tu catálogo de cajas solo trae números; normalmente son pulgadas).
    if unit == "mm":
        ud_cm = [x/10.0 for x in ud]
        return {"ok": True, "unit": "mm", "msg": f"Recibí mm. Medidas aprox en cm: {ud_cm}. Dime si tu etiqueta de cajas está en pulgadas o cm."}

    if unit == "cm":
        # no asumimos que tu catálogo está en cm o pulgadas: sugerimos pedir confirmación
        candidate = _pick(ud)
        if candidate:
            return {"ok": True, "unit": "cm", "candidate": candidate, "mode": "direct"}
        return {"ok": True, "unit": "cm", "candidate": None, "mode": "direct"}

    if unit == "in":
        candidate = _pick(ud)
        return {"ok": True, "unit": "in", "candidate": candidate, "mode": "direct"}

    # unit desconocida: damos 2 rutas (cm vs in) sin afirmar cuál es
    cand_as_in = _pick(ud)
    # si fuera cm, es muy probable que cualquiera cubra; pero no lo afirmamos: pedimos unidad
    return {"ok": True, "unit": None, "candidate_in": cand_as_in}


def describe_size(query: str) -> Optional[str]:
    """
    Intenta describir el tamaño SIN inventar:
    - Si el usuario escribe 'Caja 24x24x24' => devuelve medidas
    - Si escribe '18gal' => explica que es categoría por capacidad
    - Si escribe 'TV 50-55' => devuelve rango
    """
    q = (query or "").strip()

    # 1) Captura cajas por patrón (aunque no coincida exactamente con clave)
    m = BOX_DIM_RE.search(q)
    if m:
        kind = m.group(1)
        a, b, c = m.group(2), m.group(3), m.group(4)
        return f"{kind.title()} con medidas: {a} × {b} × {c} (según lo que escribiste)."

    # 2) Captura TVs por patrón
    m = TV_RE.search(q)
    if m:
        lo, hi = m.group(1), m.group(2)
        return f"TV por tamaño de pantalla: {lo} a {hi} pulgadas."

    # 3) Captura galones por patrón (solo capacidad)
    m = GAL_RE.search(q)
    if m:
        n = m.group(1)
        key = f"{n}gal"
        info = SIZE_KB.get(key)
        if info:
            return (
                f"{key} es una categoría por capacidad ({n} galones). "
                f"Las medidas físicas varían por marca/modelo. "
            )
        return (
            f"Es un contenedor por capacidad ({n} galones). "
            "Las medidas físicas varían por modelo. Para confirmar tamaño exacto, dime largo/ancho/alto."
        )

    # 4) Búsqueda directa por clave (si el usuario escribió igualito)
    q_norm = q.lower()
    for k, info in SIZE_KB.items():
        if k.lower() in q_norm:
            det = info.get("detalle", "")
            unidad = info.get("unidad")
            extra = f" ({unidad})" if unidad else ""
            return f"{k}: {det}{extra}"

    return None

# ======================
# TRANSFORMER + CLASIFICADOR
# ======================

print("Cargando encoder transformer...")
_encoder = SentenceTransformer(MODEL_NAME)
print("Encoder cargado:", MODEL_NAME)

# Construir dataset desde patterns
_train_texts: List[str] = []
_train_labels: List[str] = []

for intent in intents:
    tag = intent.get("tag")
    for p in intent.get("patterns", []):
        if isinstance(p, str) and p.strip():
            _train_texts.append(p.strip())
            _train_labels.append(tag)

# Entrenamiento supervisado rápido
_labeler = LabelEncoder()
y = _labeler.fit_transform(_train_labels)

X = _encoder.encode(_train_texts, normalize_embeddings=True, show_progress_bar=False)

_clf = LogisticRegression(
    max_iter=4000,
    n_jobs=None,
    class_weight="balanced"
)
_clf.fit(X, y)

# ======================
# UTILIDADES
# ======================

def _topk_probs(user_text: str, k: int = 3) -> List[Tuple[str, float]]:
    emb = _encoder.encode([user_text], normalize_embeddings=True, show_progress_bar=False)
    probs = _clf.predict_proba(emb)[0]
    idxs = np.argsort(probs)[::-1][:k]
    out = []
    for i in idxs:
        out.append((_labeler.inverse_transform([i])[0], float(probs[i])))
    return out

def _keyword_boost(user_text: str, candidates: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """
    Pequeño ajuste para desambiguar cuando hay intents parecidos.
    NO crea intents nuevos; solo reordena/ajusta ligeramente por keywords.
    """
    tl = (user_text or "").lower()
    boosted = []
    for tag, p in candidates:
        kws = keywords_map.get(tag, []) or []
        hit = 0
        for kw in kws:
            kw = (kw or "").strip().lower()
            if kw and kw in tl:
                hit += 1
        # Boost suave, para no sobre-dominarlos
        p2 = min(0.999, p + 0.03 * min(hit, 3))
        boosted.append((tag, p2))
    boosted.sort(key=lambda x: x[1], reverse=True)
    return boosted

def _pick_response(tag: str) -> str:
    resps = responses_map.get(tag) or []
    if not resps:
        return random.choice(fallback_responses)
    return random.choice(resps)

# Detección de origen México (para bloqueo de envíos nacionales)
# Solo aplica cuando el usuario indica un ORIGEN mexicano, no una pregunta de cobertura/destino
MEXICO_ORIGIN_RE = re.compile(
    r"\b(de|desde|origen|en)\s+(aguascalientes|baja\s+california|bcs|baja\s+california\s+sur|campeche|chiapas|chihuahua|cdmx|ciudad\s+de\s+méxico|df|edomex|coahuila|colima|durango|guanajuato|guerrero|hidalgo|jalisco|michoacan|michoacán|morelos|nayarit|nuevo\s+leon|nuevo\s+león|oaxaca|puebla|queretaro|querétaro|quintana\s+roo|san\s+luis|slp|sinaloa|sonora|tabasco|tamaulipas|tlaxcala|veracruz|yucatan|yucatán|zacatecas|leon|le[oó]n|guadalajara|monterrey|tijuana|juarez|juárez|toluca|torreon|torreón|reynosa|matamoros|nogales|hermosillo|culiacan|culiacán|mazatlan|mazatlán|vallarta|acapulco|cuernavaca|pachuca|saltillo|durango|mexicali|ensenada|la\s+paz|los\s+cabos|morelia|villahermosa|tuxtla|merida|mérida|zacatecas|cancun|cancún)\b",
    re.IGNORECASE
)

# Preguntas de cobertura/destino — estas NO deben activar restriccion_geografica
DESTINATION_Q_RE = re.compile(
    r"\b(a\s+qu[eé]\s+ciudades|a\s+qu[eé]\s+estados|a\s+qu[eé]\s+partes|qu[eé]\s+(estados|ciudades|destinos|lugares)\s+(atienden|cubren|tienen|manejan|llegan)|"
    r"cobertura|toda\s+la\s+rep[uú]blica|todo\s+m[eé]xico|a\s+m[eé]xico|hacen\s+env[ií]os|llegan\s+a|"
    r"ciudades\s+peque[nñ]as|norte\s+de\s+m[eé]xico|sureste|sur.*m[eé]xico|"
    r"qu[eé]\s+ciudades\s+de|ciudades\s+de\s+\w|qu[eé]\s+lugares\s+de|d[oó]nde\s+llegan\s+en|visitan\s+de|\s+de\s+\w+\s+visitan)\b",
    re.IGNORECASE
)

# Detección de DESTINO mexicano en frases de envío
# Ej: "quiero hacer un envío a Aguascalientes", "mando algo para Jalisco"
_MX_PLACES = (
    r"aguascalientes|baja\s+california\s+sur|baja\s+california|campeche|chiapas|chihuahua|"
    r"ciudad\s+de\s+m[eé]xico|cdmx|coahuila|colima|durango|guanajuato|guerrero|hidalgo|"
    r"jalisco|michoac[aá]n|morelos|nayarit|nuevo\s+le[oó]n|oaxaca|puebla|quer[eé]taro|"
    r"quintana\s+roo|san\s+luis\s+potos[ií]|san\s+luis|sinaloa|sonora|tabasco|tamaulipas|"
    r"tlaxcala|veracruz|yucat[aá]n|zacatecas|le[oó]n|guadalajara|monterrey|tijuana|"
    r"ju[aá]rez|toluca|torre[oó]n|reynosa|matamoros|nogales|hermosillo|culiac[aá]n|"
    r"mazatl[aá]n|vallarta|acapulco|cuernavaca|pachuca|saltillo|mexicali|ensenada|"
    r"la\s+paz|los\s+cabos|morelia|villahermosa|tuxtla|m[eé]rida|canc[uú]n|m[eé]xico"
)
MEXICO_DEST_RE = re.compile(
    r"\b(env[ií]o\s+a|env[ií]o\s+para|mandar?\s+a|mandar?\s+para|enviar?\s+a|enviar?\s+para|"
    r"llevar?\s+a|send\s+to|quiero\s+(?:mandar|enviar|llevar)\s+(?:\w+\s+)?(?:a|para)|"
    r"(?:a|para|hacia)\s+)(" + _MX_PLACES + r")\b",
    re.IGNORECASE
)

# Palabras clave de peso (para aclarar que no importa)
WEIGHT_RE = re.compile(r"\b(peso|pesa|kilos?|kg|libras?|lb)\b", re.IGNORECASE)

PRICE_Q_RE = re.compile(r"\b(precio|cuesta|costo|tarifa|cotiz|cotizaci[oó]n|cu[aá]nto\s+cuesta|cu[aá]nto\s+cobra|cu[aá]nto\s+cobran|a\s+cu[aá]nto|cu[aá]nto\s+vale|dame\s+una\s+idea|valor\s+del\s+env[ií]o|promoci[oó]n|descuento|oferta)\b", re.IGNORECASE)


ROUTE_Q_RE = re.compile(
    r"\b(ruta|rutas|trayecto|trayectos|recorrido|recorridos|por\s+d[oó]nde|por\s+donde|distribuyen|camino|itinerario|qu[eé]\s+ciudades\s+pasan|por\s+qu[eé]\s+ciudades|qu[eé]\s+estados\s+cubren|qu[eé]\s+ciudades\s+llegan|a\s+d[oó]nde\s+llegan)\b",
    re.IGNORECASE
)

HELP_Q_RE = re.compile(
    r"\b(otra\s+duda|otra\s+pregunta|una\s+duda\s+m[aá]s|m[aá]s\s+dudas|ay[uú]dame|me\s+ayudas|orientame|necesito\s+ayuda|me\s+pueden\s+ayudar|c[oó]mo\s+funciona|informaci[oó]n\s+sobre|sus\s+servicios|qu[eé]\s+servicios|me\s+ayuda|puedes\s+ayudarme|mandan\s+(?!joyas?|armas?|medicam|dinero|droga|mascota)|env[ií]an\s+(?!joyas?|armas?|medicam|dinero)|manejan\s+(?!joyas?))\b",
    re.IGNORECASE
)

# Preguntas de ubicación de oficinas/sucursales
UBICACION_RE = re.compile(
    r"\b(d[oó]nde\s+(est[aá]n|est[aá]\s+la|queda[n]?|puedo|se\s+ubica)|direcci[oó]n|oficina|sucursal|dejar\s+(mi\s+)?paquete|entregar\s+(el\s+)?paquete|punto\s+de\s+entrega|reciben|d[oó]nde\s+los\s+encuentro|ubicaci[oó]n|maps?\b)\b",
    re.IGNORECASE
)

# Preguntas difusas de servicio general
SERVICIO_RE = re.compile(
    r"\b(confiab(le|les)|seguros?|garant[ií]a|c[oó]mo\s+funciona|qu[eé]\s+ofrecen|qu[eé]\s+hacen|mandar\s+cosas?|enviar\s+algo|quiero\s+mandar|quiero\s+enviar|mandar\s+a\s+m[eé]xico|cosas\s+a\s+m[eé]xico|env[ií]os\s+a\s+m[eé]xico|c[oó]mo\s+le\s+hago|c[oó]mo\s+puedo\s+enviar|orientar(me)?\s+sobre|c[oó]mo\s+enviar|onda\s+con\s+los\s+precios?|qu[eé]\s+onda\s+con)\b",
    re.IGNORECASE
)

TRANSPORTE_RE = re.compile(
    r"\b(transporte|aereo|a[eé]reo|avion|avi[oó]n|tierra|terrestre|carretera|trailer|camion|cami[oó]n|por\s+aire|por\s+tierra|tipo\s+transporte|modalidad)\b",
    re.IGNORECASE
)

# Horarios de atención
HORARIOS_RE = re.compile(
    r"\b(horario|horarios|hora[s]?|a\s+qu[eé]\s+hora|cu[aá]ndo\s+(abren|cierran|atienden|trabajan)|trabajan\s+(los\s+)?(s[aá]bados?|domingos?|lunes|fines?)|d[ií]as\s+de\s+atenci[oó]n|abren|cierran|atienden|cu[aá]l\s+es\s+su\s+horario|fin\s+de\s+semana)\b",
    re.IGNORECASE
)


# Zelle y pagos digitales
ZELLE_RE = re.compile(
    r"\b(zelle|qu[eé]\s+es\s+zelle|c[oó]mo\s+funciona\s+zelle|pagar?\s+(?:con\s+)?zelle|pago\s+digital|aceptan?\s+zelle|transferencia\s+bancaria)\b",
    re.IGNORECASE
)

PAGOS_RE = re.compile(
    r"\b(m[eé]todo[s]?\s+de\s+pago|forma[s]?\s+de\s+pago|c[oó]mo\s+pago|c[oó]mo\s+se\s+paga|pagar\s+el\s+env[ií]o|aceptan?\s+(?:tarjeta|efectivo|paypal|venmo|cash))\b",
    re.IGNORECASE
)

# Comandos del bot
COMANDOS_RE = re.compile(
    r"\b(cu[aá]ntos?\s+comandos?|qu[eé]\s+comandos?|comandos?\s+tienes?|lista\s+de\s+comandos?|tienes?\s+(?:un\s+)?men[uú]|c[oó]mo\s+te\s+(?:activo|uso|hablo)|instrucciones\s+del\s+bot|palabras?\s+clave[s]?|bot\s+de\s+comandos?|necesito\s+saber\s+comandos?|c[oó]mo\s+funciona\s+este\s+chat)\b",
    re.IGNORECASE
)

# Qué te puedo preguntar / funciones del bot
QUE_PREGUNTAR_RE = re.compile(
    r"\b(qu[eé]\s+te\s+puedo\s+preguntar|sobre\s+qu[eé]\s+(?:me\s+puedes?|puedes?)\s+ayudar|qu[eé]\s+sabes?|qu[eé]\s+puedes?\s+hacer|qu[eé]\s+haces?|qu[eé]\s+puedo\s+pedirte|para\s+qu[eé]\s+sirves?|para\s+qu[eé]\s+eres?\s+[uú]til|en\s+qu[eé]\s+me\s+puedes?\s+ayudar|tus\s+funciones|tus\s+capacidades|en\s+qu[eé]\s+te\s+especializas|qu[eé]\s+temas\s+(?:manejas?|tocas?)|qu[eé]\s+tipo\s+de\s+asistencia|qu[eé]\s+puedo\s+consultar|cu[aá]l\s+es\s+tu\s+funci[oó]n|tu\s+funci[oó]n|para\s+qu[eé]\s+sirves?)\b",
    re.IGNORECASE
)

# Preguntas genéricas de tamaños/catálogo
SIZES_Q_RE = re.compile(
    r"\b(qu[eé]\s+tama[nñ]os|tama[nñ]os\s+de\s+caja|qu[eé]\s+cajas|qu[eé]\s+tipos\s+de\s+caja|tipos\s+de\s+empaque|qu[eé]\s+manejan|cat[aá]logo\s+de\s+(cajas|tama[nñ]os)|cajas\s+(disponibles|que\s+tienen)|opciones\s+de\s+(caja|empaque)|caja\s+para\s+(bici|bicicleta|moto|tv|televisi[oó]n)|cajas\s+para|tama[nñ]o\s+de\s+env[ií]o)\b",
    re.IGNORECASE
)

# Tiempo de entrega
TIEMPO_RE = re.compile(
    r"\b(cu[aá]nto\s+tarda[n]?|cu[aá]ntos?\s+d[ií]as|tiempo\s+de\s+entrega|plazo|d[ií]as\s+tarda|en\s+cu[aá]nto\s+tiempo|cu[aá]ndo\s+llega|llega\s+en|exprés|express|urgente|r[aá]pido|demoran?|duran?)\b",
    re.IGNORECASE
)

# Estados de EE.UU. NO válidos como origen — se usa para detectar frases como
# "envío de Arkansas", "si salgo de California", etc.
# NOTA: se excluyen las abreviaturas de 2 letras (me, or, la, in, ok...) porque
# generan falsos positivos con palabras comunes en español.
USA_INVALID_ORIGIN_RE = re.compile(
    r"\b(arkansas|california|florida|nevada|arizona|new\s+york|nueva\s+york|illinois|ohio|michigan|new\s+jersey|"
    r"virginia\s+occidental|virginia|washington|colorado|minnesota|oregon|louisiana|kentucky|south\s+carolina|"
    r"connecticut|utah|iowa|mississippi|kansas|nebraska|west\s+virginia|idaho|hawaii|maine|new\s+hampshire|"
    r"rhode\s+island|montana|delaware|south\s+dakota|north\s+dakota|alaska|vermont|wyoming|new\s+mexico|"
    r"missouri|indiana|maryland|north\s+carolina|pennsylvania|wisconsin|massachusetts|oklahoma)\b",
    re.IGNORECASE
)

# Estados válidos (para evitar falsos positivos)
USA_VALID_ORIGIN_RE = re.compile(
    r"\b(georgia|alabama|texas|tennessee)\b",
    re.IGNORECASE
)

# Detección de solicitudes de contacto/teléfono/asesor humano
CONTACTO_RE = re.compile(
    r"\b(tel[eé]fono|tel[eé]fonos|n[uú]mero|n[uú]meros|contacto|whatsapp|llamar|llama|marcar|"
    r"asesor|humano|persona|operador|hablar\s+con|atenci[oó]n\s+personal|encargado|"
    r"me\s+at[ieé]nde|no\s+quiero\s+(el\s+)?bot|dame\s+un\s+n[uú]mero|"
    r"p[aá]same|me\s+das\s+el|me\s+puedes\s+dar|quiero\s+llamar|quiero\s+hablar)\b|📞",
    re.IGNORECASE
)

# Métodos de pago
PAGO_RE = re.compile(
    r"\b(pagar|cobran|cobras|cobro|pagos?|formas?\s+de\s+pago|m[eé]todos?\s+de\s+pago|"
    r"tarjeta|cr[eé]dito|d[eé]bito|zelle|venmo|paypal|efectivo|d[oó]lares?|pesos?|"
    r"transferencia|dep[oó]sito|cobro\s+en\s+l[ií]nea|pago\s+en\s+l[ií]nea|cash|"
    r"acepta[n]?\s+(tarjeta|efectivo|zelle|paypal|venmo|dolares|card)|"
    r"c[oó]mo\s+se\s+paga|c[oó]mo\s+cobran|c[oó]mo\s+pago|"
    r"(qu[eé]\s+)?(m[eé]todos?|formas?)\s+(de\s+pago\s+)?(tienen|manejan|usan|aceptan|manejan|ofrecen)|"
    r"(tienen|manejan|usan|aceptan)\s+(m[eé]todos?|formas?)\s+de\s+pago|"
    r"cu[aá]les?\s+son\s+sus\s+(m[eé]todos?|formas?)\s+de\s+pago|"
    r"con\s+qu[eé]\s+pago|pago\s+con|se\s+puede\s+pagar)\b",
    re.IGNORECASE
)

# Respuesta hardcodeada de métodos de pago (no depende de intents.json)
_PAGO_RESPONSE = (
    "Manejamos los siguientes métodos de pago: 💳\n\n"
    "• 💚 Zelle\n"
    "• 🏦 Transferencia bancaria\n"
    "• 💵 Efectivo (en nuestros puntos de entrega)\n"
    "• 💳 Tarjeta de débito/crédito (consultar disponibilidad)\n\n"
    "El pago se realiza al momento de entregar el paquete en nuestro punto. "
    "Si tienes dudas sobre el método más conveniente para ti, escríbenos al:\n"
    "📲 +52 418 110 7243"
)

# Recolección / pickup — el servicio NO hace domicilio, usa puntos fijos
RECOLECCION_RE = re.compile(
    r"\b(recogen|recoger|recoge|pickup|pick\s*up|recoleccion|recolección|"
    r"van\s+a\s+(mi\s+)?(casa|domicilio|direcci[oó]n)|"
    r"(pueden\s+)?pasa[rn]?\s+por|vengan\s+a\s+recoger|"
    r"domicilio\s+a\s+recoger|a\s+domicilio\s*(para\s+)?recoger|"
    r"cobran\s+por\s+la\s+recolec|servicio\s+a\s+domicilio)\b",
    re.IGNORECASE
)

# Iniciar un pedido / envío → redirige a cotización
PEDIDO_RE = re.compile(
    r"\b(hacer\s+un\s+pedido|hago\s+un\s+pedido|iniciar\s+(un\s+)?env[ií]o|"
    r"ordenar\s+(un\s+)?env[ií]o|contratar\s+el\s+servicio|quiero\s+ordenar|"
    r"c[oó]mo\s+hago\s+un\s+pedido|c[oó]mo\s+inicio|empezar\s+un\s+env[ií]o|"
    r"quiero\s+mandar\s+algo|quiero\s+enviar\s+algo|quiero\s+hacer\s+un\s+env[ií]o)\b",
    re.IGNORECASE
)

# Incidencias post-envío (retrasos, daños, paquete equivocado)
INCIDENCIA_RE = re.compile(
    r"\b(no\s+ha\s+llegado|no\s+llega|retrasad[ao]|tardando\s+mucho|"
    r"semanas?\s+y\s+no\s+llega|llegó\s+(da[ñn]ado|roto|golpeado|mojado)|"
    r"paquete\s+(da[ñn]ado|roto|equivocado|perdido|incompleto)|"
    r"mandaron\s+el\s+paquete\s+equivocado|destinatario\s+no\s+ha\s+recibido|"
    r"no\s+recibi[oó]|problema\s+con\s+(mi\s+)?env[ií]o|queja|reclamo)\b",
    re.IGNORECASE
)

# Mención de estados EE.UU. válidos sin contexto de cotización → invitar a cotizar
ORIGEN_VALIDO_MENCIONA_RE = re.compile(
    r"\b(tengo\s+(mi\s+)?(caja|paquete|env[ií]o)\s+en|"
    r"salgo\s+de|estoy\s+en|vivo\s+en|me\s+encuentro\s+en|"
    r"env[ií]o\s+desde|mando\s+desde|env[ií]o\s+de)\s+"
    r"(georgia|alabama|texas|tennessee)\b",
    re.IGNORECASE
)

#
#
#


# ═══════════════════════════════════════════════════════════
# CIUDADES POR ESTADO — desde Destino_especiales.xlsx (ambas pestañas)
# Para flujo de cobertura: usuario pregunta por estado → bot lista ciudades
# ═══════════════════════════════════════════════════════════
CIUDADES_POR_ESTADO = {
    "Aguascalientes": ["Aguascalientes", "Calvillo", "Jesús María", "Pabellón de Arteaga", "Rincón de Romo", "San Francisco de los Romo", "Teocaltiche", "Villa Hidalgo"],
    "Baja California": ["Ensenada", "Mexicali", "Puerto Peñasco", "San Luis Río Colorado", "San Quintín", "Tijuana", "Vicente Guerrero"],
    "Baja California Sur": ["Heroica Mulegé", "La Paz", "Loreto", "Los Cabos"],
    "Campeche": ["Campeche", "Cd del Carmen"],
    "Chiapas": ["Arriaga", "Berriozábal", "Chiapas de Corzo", "Cintalapa", "Comitán de Domínguez", "Palenque", "Pijijiapan", "San Cristóbal de las Casas", "Suchiapa", "Tapachula", "Tonalá", "Tuxtla Gutiérrez"],
    "Chihuahua": ["Camargo", "Cd Cuauhtémoc", "Cd Juárez", "Chihuahua", "Delicias", "Hidalgo del Parral", "Jiménez"],
    "Ciudad de México": ["Azcapotzalco", "Benito Juárez", "Coyoacán", "Cuajimalpa de Morelos", "Iztapalapa", "La Magdalena Contreras", "Miguel Hidalgo", "Milpa Alta", "Tlalpan", "Tláhuac", "Venustiano Carranza", "Xochimilco", "Álvaro Obregón"],
    "Coahuila": ["Acuña", "Arteaga", "Castaños", "Francisco I. Madero", "Frontera", "Matamoros", "Monclova", "Nava", "Piedras Negras", "Ramos Arizpe", "Saltillo", "Torreón"],
    "Colima": ["Colima", "Manzanillo", "Villa de Álvarez"],
    "Durango": ["Durango", "Gómez Palacio", "Lerdo", "Pueblo Nuevo", "Santiago Papasquiaro"],
    "Estado de México": ["Atenco", "Atizapán de Zaragoza", "Atlacomulco", "Av. Central", "Av. Revolución", "Chalco", "Chicoloapan", "Chiconcuac", "Chimalhuacán", "Coacalco", "Coyotepec", "Cuautitlán Izcalli", "Ecatepec", "Ecatepec 2", "Huehuetoca", "Huixquilucan", "Ignacio Zaragoza", "Iztacalco", "Jilotepec", "La Paz", "Marina Nacional", "Melchor Ocampo", "Metepec", "Naucalpan", "Naucalpan de Juárez", "Nezahualcóyotl", "Nicolás Romero", "Ocoyoacac", "Palotitlán", "Soyaniquilpan de Juárez", "Temac", "Teoloyucac", "Tepetlaoxtoc", "Tepotzotlán", "Texcoco", "Tlalnepantla", "Toluca", "Tonatico", "Tultepec", "Tultitlán", "Vallejo", "Vallejo 2"],
    "Guanajuato": ["Abasolo", "Celaya", "Cortazar", "Dolores Hidalgo", "Guanajuato", "Irapuato", "León", "León aeropuerto", "León central camionera", "Moroleón", "Purísima del Rincón", "Pénjamo", "Romita", "Salvatierra", "San Diego de la Unión", "San Francisco del Rincón", "San José Iturbide", "San Luis de la Paz", "San Miguel de Allende", "Silao", "Uriangato"],
    "Guerrero": ["Acapulco", "Acapulco de Juárez", "Chilpancingo de los Bravo", "Iguala de la Independencia", "Taxco de Alarcón", "Zihuatanejo (Ixtapa)"],
    "Hidalgo": ["Actopan", "Apan", "Atitalaquia", "Atotonilco el Grande", "Cuautepec de Hinojosa", "Guerrero", "Huichapan", "Ixmiquilpan", "Juitepec", "Mineral de la Reforma", "Mixquiahuala de Juárez", "Pachuca de Soto", "Santiago Tulantepec de Lugo", "Tepeapulco", "Tepeji del Río de Ocampo", "Tizayuca", "Tlahuelilpan", "Tula de Allende", "Tulancingo de Bravo", "Yautepec"],
    "Jalisco": ["Arandas", "Atotonilco el Alto", "Autlán de Navarro", "Chapala", "Cocula", "El Salto", "Guadalajara", "Jalostitlán", "Jesús María", "Jocotepec", "La Barca", "Lagos de Moreno", "Magdalena", "Ocotlán", "Puerto Vallarta", "San Juan de los Lagos", "San Miguel el Alto", "Tala", "Tepatitlán", "Tlajomulco de Zúñiga", "Tlaquepaque", "Tonalá", "Zapopan", "Zapopan Periférico", "Zapotlanejo", "Zapotlán el Grande"],
    "Michoacán": ["Apatzingán de la Constitución", "Cd Hidalgo", "Jacona", "Jiquilpan", "La Piedad de Cabadas", "Lázaro Cárdenas", "Maravatío de Ocampo", "Morelia", "Morelia Tarimbaro", "Purépero Ichais", "Pátzcuaro", "Quiroga", "Sahuayo de Morelos", "Tarímbaro", "Uruapan", "Zacapu", "Zamora de Hidalgo"],
    "Morelos": ["Cuernavaca"],
    "Nayarit": ["Acaponeta", "Bahía de Banderas", "Compostela", "Rosa Morada", "Ruiz", "Santa María del Oro", "Santiago Ixcuintla", "Tecuala", "Tepic", "Tuxpan", "Xalisco"],
    "Nuevo León": ["Acadereita Jiménez", "Allende", "Apodaca", "García", "Gral. Escobedo", "Guadalupe", "Juárez", "Linares", "Lincoln", "Montemorelos", "Monterrey", "Monterrey Estadio", "Monterrey Universidad", "San Nicolás de la Garza", "San Pedro Garza García", "Santa Catarina", "Santa Catarina CEDIS", "Santa Catarina Paseo", "Santiago"],
    "Oaxaca": ["Asunción Nochixtlán", "Huajuapan de León", "Mazatlán", "Oaxaca de Juárez", "San Agustín Etla", "San Agustín de las Juntas", "San Andrés Huayápam", "San Antonio de la Cal", "San Jacinto Amilpas", "San Juan Bautista Tuxtepec", "San Lorenzo Cacaotepec", "San Pablo Etla", "San Sebastián Tutla", "Santa Cruz Amilpas", "Santa Cruz Xoxocotlán", "Santa Lucía del Camino", "Santa María del Tule", "Santiago Matatlán", "Tlalixtla", "Villa de Etla"],
    "Puebla": ["Acajete", "Acatlán", "Acatzingo", "Amozol", "Atilizco", "Chignahuapan", "Chinautla", "Coapiaxcla de Madero", "Cuautlancingo", "Izúcar de Matamoros", "Libres", "Puebla", "Puebla Avenida 31", "Puebla Estadio", "Quecholac", "San Jose Chiapa", "San Martín Texmelucan", "San Pedro Cholula", "San Salvador Huixcolotla", "Tecamachalco", "Tehuacán", "Tepeaca", "Teziutlán", "Zacatlán", "Zicotepec"],
    "Querétaro": ["Cadereyta de Morelos", "Colón", "Corregidora", "El Marqués", "Ezequiel Montes", "Pedro Escobedo", "Querétaro", "San Juan del Río", "Tequisquiapan"],
    "Quintana Roo": ["Cancún", "Chetumal", "Cozumel", "Playa del Carmen", "Puerto Morelos"],
    "San Luis Potosí": ["Ciudad Fernández", "Ciudad Valles", "Matehuala", "Río Verde", "San Luis Potosí", "Soledad de Graciano Sánchez"],
    "Sinaloa": ["Culiacán", "Culiacán Pedro Infante", "El Rosario", "Elota", "Escuinapa", "Guasave", "Los Mochis", "Mazatlán", "Navolato", "Salvador Alvarado"],
    "Sonora": ["Agua Prieta", "Caborca", "Cananea", "Cd Obregón", "Ciudad Obregón", "Etchojoa", "Guaymas", "Hermosillo", "Hermosillo CEDIS", "Huatabampo", "Magdalena", "Navojoa", "Nogales", "Puerto Peñasco", "San Luis Río Colorado", "Santa Ana", "Ímuris"],
    "Tabasco": ["Comalcalco", "Cunduacán", "Cárdenas", "Huimanguillo", "Jalapa", "Jalapa de Méndez", "Macuspana", "Nacajuca", "Paraíso", "Teapa", "Villahermosa"],
    "Tamaulipas": ["Altamira", "Cd Madero", "Cd Victoria", "Matamoros", "Miguel Alemán", "Nuevo Laredo", "Reynosa", "Río Bravo", "Tampico"],
    "Tlaxcala": ["Apizaco", "Chiautempan", "Huamantla", "Panotla", "Santa Isabel Xiloxoxtla", "Tepayanco", "Tetla de la Solidaridad", "Titlan de Antonio Carvajal", "Tlaxcala", "Tlaxco", "Totolac", "Xaloztoc", "Xicohtzinco", "Zacatelco"],
    "Veracruz": ["Coatzacolacos", "Veracruz", "Xalapa"],
    "Yucatán": ["Mérida", "Ticul"],
    "Zacatecas": ["Zacatecas"],
}

# Lista ordenada de estados para mostrar como opciones rápidas
ESTADOS_LIST = sorted(CIUDADES_POR_ESTADO.keys())

# Mapa normalizado para detectar el estado que el usuario escribe/selecciona
import unicodedata as _ud2
def _norm_est(s):
    s = _ud2.normalize('NFD', str(s).strip())
    return ''.join(c for c in s if _ud2.category(c) != 'Mn').lower()

ESTADO_NORM_MAP = {_norm_est(e): e for e in CIUDADES_POR_ESTADO}
# Alias comunes
ESTADO_NORM_MAP.update({
    'cdmx': 'Ciudad de México',
    'ciudad de mexico': 'Ciudad de México',
    'edomex': 'Estado de México',
    'edo mex': 'Estado de México',
    'estado de mexico': 'Estado de México',
    'nl': 'Nuevo León',
    'nuevo leon': 'Nuevo León',
    'bcs': 'Baja California Sur',
    'bc': 'Baja California',
    'qro': 'Querétaro',
    'queretaro': 'Querétaro',
    'slp': 'San Luis Potosí',
    'san luis potosi': 'San Luis Potosí',
    'gto': 'Guanajuato',
    'jalisco': 'Jalisco',
    'michoacan': 'Michoacán',
})

# Regex para detectar preguntas de cobertura general (¿a qué estados/ciudades envían?)
COBERTURA_GENERAL_RE = re.compile(
    r"\b(env[ií]an?\s+a\s+todo\s+m[eé]xico|env[ií]os?\s+nacionales?|"
    r"hacen?\s+env[ií]os?\s+(a\s+todo|nacionales?)|"
    r"a\s+todo\s+m[eé]xico|toda\s+la\s+rep[uú]blica|"
    r"cubren\s+toda\s+(la\s+)?rep[uú]blica|"
    r"a\s+qu[eé]\s+(estados?|ciudades?|partes?|lugares?)\s+(env[ií]an?|llegan?|tienen?|cubren?|hacen?)|"
    r"qu[eé]\s+(estados?|ciudades?|destinos?)\s+(cubren?|tienen?|atienden?|manejan?)|"
    r"q(ue|é)?\s+ciudades\s+tienen|"
    r"qu[eé]\s+estados\s+(llegan?|van)|"
    r"a\s+d[oó]nde\s+(env[ií]an?|llegan?|mandan?))\b",
    re.IGNORECASE
)

# Regex para detectar cuando el usuario MENCIONA un estado específico buscando cobertura
# Ej: "¿llegan a Jalisco?", "¿tienen cobertura en Michoacán?"
ESTADO_CONSULTA_RE = re.compile(
    r"\b(llegan?\s+a|env[ií]an?\s+a|tienen?\s+(cobertura\s+en|servicio\s+en)|"
    r"cubren?\s+|atienden?\s+en|hay\s+cobertura\s+en)\s+"
    r"(aguascalientes|baja\s+california\s+sur|baja\s+california|campeche|chiapas|chihuahua|"
    r"ciudad\s+de\s+m[eé]xico|cdmx|coahuila|colima|durango|guanajuato|guerrero|hidalgo|"
    r"jalisco|michoac[aá]n|morelos|nayarit|nuevo\s+le[oó]n|oaxaca|puebla|quer[eé]taro|"
    r"quintana\s+roo|san\s+luis\s+potos[ií]|sinaloa|sonora|tabasco|tamaulipas|tlaxcala|"
    r"veracruz|yucat[aá]n|zacatecas|estado\s+de\s+m[eé]xico|edomex)\b",
    re.IGNORECASE
)

# ═══════════════════════════════════════════════════════════
# EARLY-CHECKS ADICIONALES — Prioridad 1 (Diagnóstico 700 preguntas)
# ═══════════════════════════════════════════════════════════

# Despedidas / agradecimientos
DESPEDIDA_RE = re.compile(
    r"^(gracias|muchas\s+gracias|ok\s+gracias|gracias\s+por\s+(todo|la\s+info|la\s+ayuda|su\s+atenci[oó]n)|"
    r"adi[oó]s|hasta\s+(luego|pronto|la\s+vista)|chau|chao|bye[\s!]*|nos\s+vemos|"
    r"cu[ií]date|que\s+tengas\s+buen|buen\s+d[ií]a\s+para|ya\s+entend[ií]|ok\s+ya\s+me\s+quedo|"
    r"perfecto\s+gracias|listo\s+gracias|excelente\s+gracias|de\s+nada|con\s+mucho\s+gusto)[\s!.]*$",
    re.IGNORECASE
)

# Cotización / precio — sin ya incluir la pregunta de measures
COTIZAR_RE = re.compile(
    r"\b(cu[aá]nto\s+(cuesta|cobran|sale|es\s+el\s+precio|est[aá])|"
    r"qu[eé]\s+precio|a\s+cu[aá]nto\s+sale|"
    r"tarifa[s]?|flete|costo\s+del?\s+env[ií]o|precio\s+(del?|de\s+la\s+)\s*(env[ií]o|caja|barril|gal[oó]n|tamba?or)|"
    r"cotizaci[oó]n|quiero\s+(cotizar|saber\s+el\s+precio|un\s+precio)|"
    r"me\s+(dan\s+(un\s+)?(precio|cotizaci[oó]n)|pueden\s+dar\s+precio)|"
    r"cu[aá]nto\s+(me\s+cobran|cobran\s+por|sale\s+mandar|cuesta\s+mandar|costaria|costará)|"
    r"precio\s+(desde|para|de)\s+\w|"
    r"cu[aá]nto\s+(para|por)\s+(una?|el?\s+)\w|"
    r"tienen\s+promocion(es)?|hay\s+descuento|"
    r"cobran\s+por\s+(kilo|peso|volumen|medidas?|tama[nñ]o))",
    re.IGNORECASE
)

# Rastreo / tracking
RASTREO_RE = re.compile(
    r"\b(d[oó]nde\s+(est[aá]|anda)\s+(mi\s+)?(paquete|caja|env[ií]o|pedido)|"
    r"tracking|rastrear?|rastreo|n[uú]mero\s+de\s+gu[ií]a|gu[ií]a\s+de\s+rastreo|"
    r"cu[aá]ndo\s+llega|estatus\s+(del?|de\s+mi)\s*(paquete|env[ií]o|pedido|caja)|"
    r"c[oó]mo\s+(rastreo|sigo|le\s+doy\s+seguimiento|consulto)\s+(mi\s+)?(env[ií]o|caja|paquete|pedido)|"
    r"(ya|cuándo)\s+(lleg[oó]|entregaron|entreg[oó])\s+mi\s+(paquete|caja)|"
    r"qu[eé]\s+pas[oó]\s+con\s+mi\s+env[ií]o|"
    r"en\s+qu[eé]\s+etapa\s+va|c[oó]mo\s+me\s+(avisan|informan|notifican)|"
    r"me\s+mandan\s+notificaci[oó]n|puedo\s+ver\s+d[oó]nde\s+va|"
    r"tienen\s+(app|aplicaci[oó]n|sistema)\s+(para\s+)?rastrear|"
    r"no\s+me\s+ha\s+llegado\s+nada|no\s+llega\s+mi\s+paquete)",
    re.IGNORECASE
)

# Tiempos de entrega
TIEMPO_RE = re.compile(
    r"\b(cu[aá]nto\s+tarda[n]?|cu[aá]ntos\s+d[ií]as\s+(tarda[n]?|demora[n]?)|"
    r"en\s+cu[aá]ntos\s+d[ií]as|cu[aá]nto\s+demora[n]?|tiempos?\s+de\s+entrega|"
    r"en\s+cu[aá]nto\s+llega|d[ií]as\s+h[aá]biles|cu[aá]ndo\s+llega\s+si\s+mando|"
    r"si\s+mando\s+(hoy|ahorita|ahora)\s+cu[aá]ndo\s+llega|"
    r"hay\s+env[ií]o\s+(express|exprés|urgente|r[aá]pido)|env[ií]o\s+urgente|"
    r"son\s+r[aá]pidos?|demoran\s+mucho|cu[aá]nto\s+tarda\s+(de|desde)|"
    r"tiempo\s+estimado|m[ií]nimo\s+cu[aá]ntos\s+d[ií]as|m[aá]ximo\s+cu[aá]ntos)",
    re.IGNORECASE
)

# Artículos que SÍ se pueden enviar
ARTICULOS_PERMITIDOS_RE = re.compile(
    r"\b(qu[eé]\s+(puedo\s+(enviar|mandar)|cosas?\s+(mandan?|env[ií]an?)|tipo\s+de\s+(paquetes?|art[ií]culos?)\s+aceptan?)|"
    r"(puedo\s+)?(mandar|enviar|mandan?)\s+(ropa|zapatos?|tenis|calzado|herramientas?|electr[oó]nicos?|electrodom[eé]sticos?|muebles?|bicicleta[s]?|juguetes?|art[ií]culos?\s+del\s+hogar)|"
    r"aceptan?\s+(ropa|herramientas?|electr[oó]nicos?|muebles?|televisiones?|bici[s]?)|"
    r"qu[eé]\s+(m[aá]s\s+)?aceptan|art[ií]culos?\s+permitidos?|qu[eé]\s+tipo\s+de\s+mercanc[ií]a)",
    re.IGNORECASE
)

# Artículos prohibidos / restricciones — EARLY (Keywords críticas)
ARTICULOS_PROHIBIDOS_RE = re.compile(
    r"\b(oro|plata|joyas?|dinero|efectivo|cash|armas?|pistolas?|municiones|drogas|estupefacientes|"
    r"animales?|perros?|gatos?|aves|comida|carne|perecederos|medicamentos?|medicinas?|"
    r"alcohol|tequila|mezcal|cigarros?|tabaco|l[ií]quidos?|perfumes?|explosivos?|fuegos\s+artificiales|"
    r"oro\s+o\s+plata|plata\s+u\s+oro)\b|"
    r"\b(qu[eé]\s+(no\s+)?puedo\s+(enviar|mandar)|qu[eé]\s+(art[ií]culos?|cosas?)\s+no\s+aceptan?|"
    r"qu[eé]\s+(est[aá]\s+)?prohibido|art[ií]culos?\s+prohibidos?|qu[eé]\s+no\s+mandan?|"
    r"qu[eé]\s+restricciones?\s+tienen|hay\s+art[ií]culos?\s+que\s+no|"
    r"no\s+aceptan?\s+(comida|medicamentos?|joyas?|armas?))",
    re.IGNORECASE
)

# Aduana / revisión de paquetes
ADUANA_RE = re.compile(
    r"\b(abren?\s+(los\s+)?paquetes?|revisan?\s+(el\s+)?(contenido|paquetes?|cajas?)|"
    r"aduana|pasa\s+por\s+aduana|revisi[oó]n\s+de\s+(paquetes?|cajas?)|inspeccionan?|"
    r"puede\s+quedar\s+retenido|retenido\s+en\s+aduana|"
    r"cobran?\s+impuestos?|hay\s+que\s+pagar\s+impuesto|pagar\s+aduanas?|"
    r"proceso\s+de\s+aduana|problemas?\s+con\s+aduana|requisan?)",
    re.IGNORECASE
)

# Cambio de dirección de entrega — EARLY
CAMBIO_DIR_RE = re.compile(
    r"\b(puedo\s+(cambiar|modificar|actualizar|corregir)\s+(la\s+)?direcci[oó]n|"
    r"me\s+equivoqu[eé]\s+de\s+direcci[oó]n|cambio\s+de\s+domicilio|"
    r"la\s+direcci[oó]n\s+est[aá]s?\s+mal|puse\s+mal\s+la\s+direcci[oó]n|"
    r"necesito\s+(modificar|cambiar)\s+(la\s+)?entrega|"
    r"puede\s+ir\s+a\s+otra\s+direcci[oó]n|cambiar\s+el\s+destinatario|"
    r"destinatario\s+(se\s+mud[oó]|ya\s+no\s+vive\s+ah[ií])|"
    r"entregaron\s+en\s+direcci[oó]n\s+incorrecta)",
    re.IGNORECASE
)

# Identidad del bot y procedencia — EARLY
IDENTIDAD_BOT_RE = re.compile(
    r"\b(quien\s+eres|quien\s+eres\s+tu|que\s+eres|como\s+te\s+llamas|cual\s+es\s+tu\s+nombre|"
    r"eres\s+(un\s+)?(bot|robot|ia|inteligencia\s+artificial|chatgpt|asistente)|"
    r"eres\s+humano|estoy\s+hablando\s+con\s+una\s+persona|me\s+estas?\s+atendiendo\s+una\s+persona|"
    r"quien\s+me\s+estas?\s+atendiendo|eres\s+de\s+(google|openai|microsoft)|"
    r"superpacky|como\s+te\s+crearon|quien\s+te\s+programo)",
    re.IGNORECASE
)

# Preguntas sobre historia/fundación de la empresa — EARLY
FUNDACION_RE = re.compile(
    r"\b(cu[aá]ndo\s+(se\s+)?(fund[oó]|naci[oó]|empez[oó]|inici[oó]|abri[oó]|crearon|arrancaron)|"
    r"desde\s+cu[aá]ndo\s+(existen?|operan?|trabajan?|est[aá]n?|dan?\s+servicio|brindan?|est[aá]n?\s+activos?)|"
    r"desde\s+qu[eé]\s+a[nñ]o\s+(existen?|operan?|trabajan?)|"
    r"en\s+qu[eé]\s+a[nñ]o\s+(abrieron|iniciaron|empezaron|arrancaron|se\s+cre[oó])|"
    r"son\s+empresa\s+nueva|son\s+nuevos|cu[aá]ntos?\s+a[nñ]os?\s+(tienen?|llevan?|cargan?)|"
    r"qu[eé]\s+trayectoria|qu[eé]\s+tan\s+(viej[ao]|antigua)|antigü?edad|cu[aá]l\s+es\s+su\s+historia|"
    r"historia\s+de\s+la\s+empresa|hace\s+cu[aá]nto\s+(empezaron|iniciaron|abrieron|existen?)|"
    r"tienen?\s+experiencia|a[nñ]o\s+de\s+(fundaci[oó]n|inicio)|fundaci[oó]n\s+de\s+la\s+empresa)\b",
    re.IGNORECASE
)

# Rastreo / Tracking / Guía — EARLY
RASTREO_ESTRICTO_RE = re.compile(
    r"\b(d[oó]nde\s+est[aá]\s+mi\s+(paquete|caja|env[ií]o)|rastr[eo][o]|tracking|n[uú]mero\s+de\s+gu[ií]a|"
    r"estatus\s+de\s+mi\s+(env[ií]o|paquete)|d[oó]nde\s+va\s+mi\s+caja|"
    r"rastrear\s+mi\s+gu[ií]a|seguimiento\s+de\s+mi\s+env[ií]o)",
    re.IGNORECASE
)

# Puntos de entrega / recolección fijos en EE.UU.
# Los 4 estados de ORIGEN: Georgia, Texas, Tennessee, Alabama
# Los 2 puntos FÍSICOS: Dalton GA y Little Rock AR
PUNTOS_USA_RE = re.compile(
    r"\b(d[oó]nde\s+(est[aá]n?\s+en\s+)?(estados?\s+unidos|usa|ee\.?\s*uu\.?)|"
    r"d[oó]nde\s+(puedo\s+)?(dejar|entregar|llevar)\s+(mi\s+)?(paquete|caja)|"
    r"puntos?\s+de\s+(entrega|recolecci[oó]n)\s+en\s+(usa|estados?\s+unidos?)|"
    r"tienen?\s+punto\s+en\s+(nashville|houston|dallas|miami|chicago|orlando|"
    r"charlotte|memphis|knoxville|louisville|cincinnati|raleigh|columbia)|"
    r"a\s+d[oó]nde\s+llevo\s+(el|mi)\s*(paquete|caja)|"
    r"c[oó]mo\s+(hago\s+llegar|entrego|llevo)\s+(mi\s+)?(caja|paquete)|"
    r"en\s+qu[eé]\s+direcci[oó]n\s+dejo|punto\s+de\s+entrega\s+en\s+(georgia|alabama|texas|tennessee)|"
    r"tienen?\s+(bodega|sucursal|punto)\s+en\s+(georgia|alabama|texas|tennessee))",
    re.IGNORECASE
)

# Respuesta de puntos USA hardcodeada (incluye los 4 estados de origen + 2 puntos físicos)
_PUNTOS_USA_RESPONSE = (
    "Recibimos paquetes desde 4 estados de EE.UU.: 🇺🇸\n\n"
    "🏠 Estados de origen:\n"
    "• Georgia • Texas • Tennessee • Alabama\n\n"
    "📦 Puntos físicos de entrega:\n"
    "📍 Dalton, Georgia — a un costado de Tienda Mexicana Talpa\n"
    "📍 Little Rock, Arkansas — contacta por WhatsApp para dirección exacta\n\n"
    "Lleva tu paquete al punto más cercano. ¿Necesitas la dirección exacta o quieres cotizar? 😊"
)

# ═══════════════════════════════════════════════════════════
# EARLY-CHECKS — salidas, origen alterno, envío a ciudad
# ═══════════════════════════════════════════════════════════

# Fechas de salida / viajes / próxima salida
SALIDA_RE = re.compile(
    r"\b(cua?ndo\s+(salen|hacen\s+salida|sale\s+el\s+(camion|trailer|viaje|envio|proximo))|"
    r"fecha[s]?\s+de\s+(salida|envio|viaje)|cua?ndo\s+tienen\s+salida|"
    r"proxima\s+salida|proximo\s+viaje|proximo\s+envio|"
    r"cada\s+cua?nt[oa]\s+(salen|envian|hacen\s+viaje|mandan)|"
    r"cua?ndo\s+es\s+la\s+(siguiente|proxima)\s+(salida|corrida)|"
    r"que\s+dias\s+salen|que\s+fechas\s+tienen|"
    r"tienen\s+salida\s+(esta\s+semana|este\s+mes|pronto)|"
    r"cua?ndo\s+tienen\s+viaje|cua?ndo\s+salen\s+de|"
    r"hay\s+salida\s+(hoy|ma[ñnan]a|esta\s+semana)|"
    r"cua?ndo\s+parten|dia\s+de\s+salida|horario\s+de\s+salida|"
    r"cuando\s+tienen\s+salida)",
    re.IGNORECASE
)

_SALIDA_RESPONSE = (
    "Las fechas de salida pueden variar según la ruta y la carga. 🚛\n\n"
    "Para información actualizada sobre próximas salidas, "
    "favor de comunicarte directamente al:\n\n"
    "📲 +52 418 110 7243\n\n"
    "Ahí te darán los detalles exactos de fechas y rutas. 😊"
)

# Envío desde otro estado (no GA/TX/TN/AL)
ORIGEN_OTRO_RE = re.compile(
    r"\b(haces?\s+env[ií]os?\s+(de|desde)\s+otro\s+(estado|lugar|punto|ciudad)|"
    r"env[ií]an?\s+desde\s+otro\s+estado|"
    r"(puedo\s+)?(mandar|enviar|mandan|env[ií]an)\s+(de|desde)\s+(california|florida|new\s+york|"
    r"north\s+carolina|carolina\s+del\s+norte|illinois|ohio|indiana|michigan|"
    r"virginia|washington|oregon|arizona|colorado|minnesota|wisconsin|"
    r"missouri|kentucky|maryland|pennsylvania|connecticut|louisiana|oklahoma|"
    r"nevada|iowa|utah|kansas|nebraska|new\s+jersey|nuevo\s+jersey|"
    r"massachusetts|otro\s+estado|otra\s+ciudad|otro\s+lugar)|"
    r"(tienen|hay)\s+(punto|cobertura|env[ií]o)\s+en\s+(california|florida|new\s+york|"
    r"north\s+carolina|illinois|ohio|indiana|michigan|virginia|washington|"
    r"oregon|arizona|colorado|otro\s+estado))",
    re.IGNORECASE
)

_ORIGEN_OTRO_RESPONSE = (
    "Actualmente solo hacemos envíos desde los siguientes puntos: 📦\n\n"
    "🇺🇸 Georgia • Texas • Tennessee • Alabama\n\n"
    "Sin embargo, nuestros transportistas pasan por algunas rutas adicionales "
    "y podríamos ayudarte según tu ubicación.\n\n"
    "Contáctanos directamente para ver si es posible realizar tu envío:\n"
    "📲 (706) 980 88 89\n"
    "📲 +52 418 110 7243\n\n"
    "¡Con gusto te orientamos! 😊"
)

# Regex para detectar "quiero hacer envío a [ciudad]", "mandar a [ciudad]", "enviar a [ciudad]"
ENVIO_A_CIUDAD_RE = re.compile(
    r"(?:quiero|quisiera|deseo|necesito|me\s+gustar[ií]a|puedo|voy\s+a|vamos\s+a)?\s*"
    r"(?:hacer|realizar)?\s*"
    r"(?:un\s+)?(?:env[ií]o|mandar?|enviar|paquete)\s+"
    r"(?:a|hacia|para|con\s+destino\s+a|al?\s+)\s*"
    r"(.+?)[\s?.!]*$",
    re.IGNORECASE
)

# ═══════════════════════════════════════════════════════════
# MAPEO DE RUTAS USA POR TELÉFONO — Segmentación por zona
# ═══════════════════════════════════════════════════════════

# Diccionario de relación Ciudad/Estado -> Teléfono
_USA_RUTAS_MAP = {
    # Grupo 1 (Termina en 8889)
    'dalton': '706 980 88 89', 'calhoun': '706 980 88 89', 'calahoun': '706 980 88 89',
    'rome': '706 980 88 89', 'atlanta': '706 980 88 89', 'seneca': '706 980 88 89',
    'sur carolina': '706 980 88 89', 'birmingham': '706 980 88 89', 'birminham': '706 980 88 89',
    'houston': '706 980 88 89', 'el campo': '706 980 88 89',

    # Grupo 2 (Termina en 0331)
    'san antonio': '512 705 03 31', 'amarillo': '512 705 03 31', 'oklahoma': '512 705 03 31',
    'tulsa': '512 705 03 31', 'wichita': '512 705 03 31', 'kansas city': '512 705 03 31',
    'joplin': '512 705 03 31', 'springfield': '512 705 03 31', 'branson': '512 705 03 31',
    'denver': '512 705 03 31', 'aurora': '512 705 03 31', 'colorado springs': '512 705 03 31',
    'fort smith': '512 705 03 31', 'rogers': '512 705 03 31', 'springdale': '512 705 03 31',
    'colorado': '512 705 03 31', 'kansas': '512 705 03 31', 'missouri': '512 705 03 31',

    # Grupo 3 (Termina en 1711)
    'little rock': '706 260 17 11', 'norte de little rock': '706 260 17 11', 
    'stuttgart': '706 260 17 11', 'memphis': '706 260 17 11', 'jackson': '706 260 17 11', 
    'humboldt': '706 260 17 11', 'nashville': '706 260 17 11', 'bowling green': '706 260 17 11', 
    'louisville': '706 260 17 11', 'cincinnati': '706 260 17 11', 'columbus': '706 260 17 11', 
    'chattanooga': '706 260 17 11', 'knoxville': '706 260 17 11', 'texarkana': '706 260 17 11', 
    'mount pleasant': '706 260 17 11', 'elgin': '706 260 17 11', 'tennessee': '706 260 17 11', 
    'kentucky': '706 260 17 11', 'ohio': '706 260 17 11',

    # Conflictos (aparecen en múltiples grupos)
    'austin': '512 705 03 31 o al 706 260 17 11',
    'dallas': '512 705 03 31 o al 706 260 17 11',
    'fort worth': '512 705 03 31 o al 706 260 17 11',
}

# Regex para detectar estas menciones
USA_RUTAS_RE = re.compile(
    r"\b(dalton|calhoun|calahoun|rome|atlanta|seneca|birmingham|birminham|houston|el\s+campo|"
    r"austin|san\s+antonio|dallas|fort\s+worth|amarillo|oklahoma|tulsa|wichita|kansas\s+city|"
    r"joplin|springfield|branson|denver|aurora|colorado\s+springs|fort\s+smith|rogers|springdale|"
    r"little\s+rock|stuttgart|memphis|jackson|humboldt|nashville|bowling\s+green|louisville|"
    r"cincinnati|columbus|chattanooga|knoxville|texarkana|mount\s+pleasant|elgin|"
    r"georgia|alabama|texas|tennessee|south\s+carolina|kansas|missouri|colorado|kentucky|ohio|arkansas)\b",
    re.IGNORECASE
)

# ═══════════════════════════════════════════════════════════
# DESTINOS ESPECIALES — Domicilio vs Ocurre (Destino_especiales.xlsx)
# Clave: nombre_ciudad_lower  →  (nombre_display, estado_display)
# ═══════════════════════════════════════════════════════════
import unicodedata as _ud

def _norm_ciudad(s):
    """Normaliza texto: minúsculas + quita acentos para comparaciones flexibles."""
    s = _ud.normalize('NFD', str(s).strip())
    s = ''.join(c for c in s if _ud.category(c) != 'Mn')
    return s.lower()

DESTINOS_DOMICILIO = {
    'aguascalientes': ('Aguascalientes', 'Aguascalientes'),
    'calvillo': ('Calvillo', 'Aguascalientes'),
    'jesús maría': ('Jesús María', 'Jalisco'),
    'pabellón de arteaga': ('Pabellón De Arteaga', 'Aguascalientes'),
    'villa hidalgo': ('Villa Hidalgo', 'Aguascalientes'),
    'teocaltiche': ('Teocaltiche', 'Aguascalientes'),
    'rincón de romos': ('Rincón De Romos', 'Aguascalientes'),
    'san francisco de los romo': ('San Francisco De Los Romo', 'Aguascalientes'),
    'ensenada': ('Ensenada', 'Baja California'),
    'mexicali': ('Mexicali', 'Baja California'),
    'san luis río colorado': ('San Luis Río Colorado', 'Sonora'),
    'puerto peñasco': ('Puerto Peñasco', 'Baja California'),
    'vicente guerrero': ('Vicente Guerrero', 'Baja California'),
    'san quintín': ('San Quintín', 'Baja California'),
    'tijuana': ('Tijuana', 'Baja California'),
    'cadereyta de morelos': ('Cadereyta De Morelos', 'Querétaro'),
    'colón': ('Colón', 'Querétaro'),
    'corregidora': ('Corregidora', 'Querétaro'),
    'el marqués': ('El Marqués', 'Querétaro'),
    'ezequiel montes': ('Ezequiel Montes', 'Querétaro'),
    'pedro escobedo': ('Pedro Escobedo', 'Querétaro'),
    'querétaro': ('Querétaro', 'Querétaro'),
    'san juan del río': ('San Juan Del Río', 'Querétaro'),
    'tequisquiapan': ('Tequisquiapan', 'Querétaro'),
    'cárdenas': ('Cárdenas', 'Tabasco'),
    'comalcalco': ('Comalcalco', 'Tabasco'),
    'cunduacán': ('Cunduacán', 'Tabasco'),
    'huimanguillo': ('Huimanguillo', 'Tabasco'),
    'jalapa': ('Jalapa', 'Tabasco'),
    'jalapa de méndez': ('Jalapa De Méndez', 'Tabasco'),
    'macuspana': ('Macuspana', 'Tabasco'),
    'nacajuca': ('Nacajuca', 'Tabasco'),
    'paraíso': ('Paraíso', 'Tabasco'),
    'teapa': ('Teapa', 'Tabasco'),
    'villahermosa': ('Villahermosa', 'Tabasco'),
    'la paz': ('La Paz', 'Edo Mex'),
    'loreto': ('Loreto', 'Baja California Sur'),
    'los cabos': ('Los Cabos', 'Baja California Sur'),
    'heroica mulegé': ('Heroica Mulegé', 'Baja California Sur'),
    'campeche': ('Campeche', 'Campeche'),
    'cd del carmen': ('Cd Del Carmen', 'Campeche'),
    'arriaga': ('Arriaga', 'Chiapas'),
    'berriozábal': ('Berriozábal', 'Chiapas'),
    'chiapas de corzo': ('Chiapas De Corzo', 'Chiapas'),
    'cintalapa': ('Cintalapa', 'Chiapas'),
    'comitán de domínguez': ('Comitán De Domínguez', 'Chiapas'),
    'palenque': ('Palenque', 'Chiapas'),
    'pijijiapan': ('Pijijiapan', 'Chiapas'),
    'san cristóbal de las casas': ('San Cristóbal De Las Casas', 'Chiapas'),
    'suchiapa': ('Suchiapa', 'Chiapas'),
    'tapachula': ('Tapachula', 'Chiapas'),
    'tonalá': ('Tonalá', 'Jalisco'),
    'tuxtla gutiérrez': ('Tuxtla Gutiérrez', 'Chiapas'),
    'altamira': ('Altamira', 'Tamaulipas'),
    'cd madero': ('Cd Madero', 'Tamaulipas'),
    'cd victoria': ('Cd Victoria', 'Tamaulipas'),
    'matamoros': ('Matamoros', 'Coahuila'),
    'miguel alemán': ('Miguel Alemán', 'Tamaulipas'),
    'nuevo laredo': ('Nuevo Laredo', 'Tamaulipas'),
    'reynosa': ('Reynosa', 'Tamaulipas'),
    'río bravo': ('Río Bravo', 'Tamaulipas'),
    'tampico': ('Tampico', 'Tamaulipas'),
    'allende': ('Allende', 'Nuevo León'),
    'apodaca': ('Apodaca', 'Nuevo León'),
    'acadereita jiménez': ('Acadereita Jiménez', 'Nuevo León'),
    'garcía': ('García', 'Nuevo León'),
    'gral escobedo': ('Gral Escobedo', 'Nuevo León'),
    'guadalupe': ('Guadalupe', 'Nuevo León'),
    'san pedro garza garcía': ('San Pedro Garza García', 'Nuevo León'),
    'sta catarina': ('Sta Catarina', 'Nuevo León'),
    'juárez': ('Juárez', 'Nuevo León'),
    'linares': ('Linares', 'Nuevo León'),
    'montemorelos': ('Montemorelos', 'Nuevo León'),
    'monterrey': ('Monterrey', 'Nuevo León'),
    'san nicolás de los garza': ('San Nicolás De Los Garza', 'Nuevo León'),
    'santiago': ('Santiago', 'Nuevo León'),
    'camargo': ('Camargo', 'Chihuahua'),
    'cd cuauhtémoc': ('Cd Cuauhtémoc', 'Chihuahua'),
    'cd juárez': ('Cd Juárez', 'Chihuahua'),
    'chihuahua': ('Chihuahua', 'Chihuahua'),
    'delicias': ('Delicias', 'Chihuahua'),
    'hidalgo del parral': ('Hidalgo Del Parral', 'Chihuahua'),
    'jiménez': ('Jiménez', 'Chihuahua'),
    'álvaro obregón': ('Álvaro Obregón', 'Ciudad de México'),
    'atzcapozalco': ('Atzcapozalco', 'Ciudad de México'),
    'benito juárez': ('Benito Juárez', 'Ciudad de México'),
    'coyoacán': ('Coyoacán', 'Ciudad de México'),
    'cuajimalpa de morelos': ('Cuajimalpa De Morelos', 'Ciudad de México'),
    'iztapalapa': ('Iztapalapa', 'Ciudad de México'),
    'magdalena contreras': ('Magdalena Contreras', 'Ciudad de México'),
    'miguel hidalgo': ('Miguel Hidalgo', 'Ciudad de México'),
    'milpa alta': ('Milpa Alta', 'Ciudad de México'),
    'tláhuac': ('Tláhuac', 'Ciudad de México'),
    'tlalpan': ('Tlalpan', 'Ciudad de México'),
    'venustiano carranza': ('Venustiano Carranza', 'Ciudad de México'),
    'xochimilco': ('Xochimilco', 'Ciudad de México'),
    'culiacán': ('Culiacán', 'Sinaloa'),
    'el rosario': ('El Rosario', 'Sinaloa'),
    'elota': ('Elota', 'Sinaloa'),
    'escuinapa': ('Escuinapa', 'Sinaloa'),
    'guasave': ('Guasave', 'Sinaloa'),
    'los mochis': ('Los Mochis', 'Sinaloa'),
    'mazatlán': ('Mazatlán', 'Sinaloa'),
    'navolato': ('Navolato', 'Sinaloa'),
    'salvador alvarado': ('Salvador Alvarado', 'Sinaloa'),
    'acuña': ('Acuña', 'Coahuila'),
    'arteaga': ('Arteaga', 'Coahuila'),
    'castaños': ('Castaños', 'Coahuila'),
    'francisco i madero': ('Francisco I Madero', 'Coahuila'),
    'frontera': ('Frontera', 'Coahuila'),
    'monclova': ('Monclova', 'Coahuila'),
    'nava': ('Nava', 'Coahuila'),
    'piedras negras': ('Piedras Negras', 'Coahuila'),
    'ramos arizpe': ('Ramos Arizpe', 'Coahuila'),
    'saltillo': ('Saltillo', 'Coahuila'),
    'torreón': ('Torreón', 'Coahuila'),
    'villa de álvarez': ('Villa De Álvarez', 'Colima'),
    'durango': ('Durango', 'Durango'),
    'gómez palacio': ('Gómez Palacio', 'Durango'),
    'lerdo': ('Lerdo', 'Durango'),
    'pueblo nuevo': ('Pueblo Nuevo', 'Durango'),
    'santiago papasquiaro': ('Santiago Papasquiaro', 'Durango'),
    'titlan de antonio car': ('Titlan De Antonio Car', 'Tlaxcala'),
    'apizaco': ('Apizaco', 'Tlaxcala'),
    'chiautempan': ('Chiautempan', 'Tlaxcala'),
    'huamantla': ('Huamantla', 'Tlaxcala'),
    'panotla': ('Panotla', 'Tlaxcala'),
    'tepeyanco': ('Tepeyanco', 'Tlaxcala'),
    'tetla de la solidaridad': ('Tetla De La Solidaridad', 'Tlaxcala'),
    'tlaxcala': ('Tlaxcala', 'Tlaxcala'),
    'tlaxco': ('Tlaxco', 'Tlaxcala'),
    'totolac': ('Totolac', 'Tlaxcala'),
    'xaloztoc': ('Xaloztoc', 'Tlaxcala'),
    'xicohtzinco': ('Xicohtzinco', 'Tlaxcala'),
    'zacatelco': ('Zacatelco', 'Tlaxcala'),
    'atenco': ('Atenco', 'Edo Mex'),
    'atizapán de zaragoza': ('Atizapán De Zaragoza', 'Edo Mex'),
    'atlacomulco': ('Atlacomulco', 'Edo Mex'),
    'chalco': ('Chalco', 'Edo Mex'),
    'chicoloapan': ('Chicoloapan', 'Edo Mex'),
    'chiconcuan': ('Chiconcuan', 'Edo Mex'),
    'chimalhuacán': ('Chimalhuacán', 'Edo Mex'),
    'coacalco': ('Coacalco', 'Edo Mex'),
    'coyotepec': ('Coyotepec', 'Edo Mex'),
    'cuautitlán izcalli': ('Cuautitlán Izcalli', 'Edo Mex'),
    'ecatepec': ('Ecatepec', 'Edo Mex'),
    'huehuetoca': ('Huehuetoca', 'Edo Mex'),
    'huixquilucan': ('Huixquilucan', 'Edo Mex'),
    'jilotepec': ('Jilotepec', 'Edo Mex'),
    'melchor ocampo': ('Melchor Ocampo', 'Edo Mex'),
    'metepec': ('Metepec', 'Edo Mex'),
    'naucalpan de juárez': ('Naucalpan De Juárez', 'Edo Mex'),
    'nezahualcóyotl': ('Nezahualcóyotl', 'Edo Mex'),
    'nicolás romero': ('Nicolás Romero', 'Edo Mex'),
    'ocoyoacac': ('Ocoyoacac', 'Edo Mex'),
    'palotitlán': ('Palotitlán', 'Edo Mex'),
    'oyaniquilpan de juárez': ('Oyaniquilpan De Juárez', 'Edo Mex'),
    'temac': ('Temac', 'Edo Mex'),
    'teoloyucan': ('Teoloyucan', 'Edo Mex'),
    'tepotzotlán': ('Tepotzotlán', 'Edo Mex'),
    'tepetlaoxtoc': ('Tepetlaoxtoc', 'Edo Mex'),
    'texcoco': ('Texcoco', 'Edo Mex'),
    'tlalnepantla': ('Tlalnepantla', 'Edo Mex'),
    'toluca': ('Toluca', 'Edo Mex'),
    'tonatico': ('Tonatico', 'Edo Mex'),
    'tultepec': ('Tultepec', 'Edo Mex'),
    'tultitlán': ('Tultitlán', 'Edo Mex'),
    'ciudad fernández': ('Ciudad Fernández', 'San Luis Potosí'),
    'ciudad valles': ('Ciudad Valles', 'San Luis Potosí'),
    'matehuala': ('Matehuala', 'San Luis Potosí'),
    'río verde': ('Río Verde', 'San Luis Potosí'),
    'san luis potosí': ('San Luis Potosí', 'San Luis Potosí'),
    'soledad de graciano sánchez': ('Soledad De Graciano Sánchez', 'San Luis Potosí'),
    'cancún': ('Cancún', 'Quintana Roo'),
    'cozumel': ('Cozumel', 'Quintana Roo'),
    'chetumal': ('Chetumal', 'Quintana Roo'),
    'playa del carmen': ('Playa Del Carmen', 'Quintana Roo'),
    'puerto morelos': ('Puerto Morelos', 'Quintana Roo'),
    'abasolo': ('Abasolo', 'Guanajuato'),
    'celaya': ('Celaya', 'Guanajuato'),
    'cortazar': ('Cortazar', 'Guanajuato'),
    'dolores hidalgo': ('Dolores Hidalgo', 'Guanajuato'),
    'guanajuato': ('Guanajuato', 'Guanajuato'),
    'irapuato': ('Irapuato', 'Guanajuato'),
    'león': ('León', 'Guanajuato'),
    'moroleón': ('Moroleón', 'Guanajuato'),
    'pénjamo': ('Pénjamo', 'Guanajuato'),
    'purísima del rincón': ('Purísima Del Rincón', 'Guanajuato'),
    'romita': ('Romita', 'Guanajuato'),
    'salvatierra': ('Salvatierra', 'Guanajuato'),
    'san francisco del rincón': ('San Francisco Del Rincón', 'Guanajuato'),
    'san josé iturbide': ('San José Iturbide', 'Guanajuato'),
    'san luis de la paz': ('San Luis De La Paz', 'Guanajuato'),
    'san miguel de allende': ('San Miguel De Allende', 'Guanajuato'),
    'silao': ('Silao', 'Guanajuato'),
    'uriangato': ('Uriangato', 'Guanajuato'),
    'arandas': ('Arandas', 'Jalisco'),
    'atotonilco el alto': ('Atotonilco El Alto', 'Jalisco'),
    'autlán de navarro': ('Autlán De Navarro', 'Jalisco'),
    'chapala': ('Chapala', 'Jalisco'),
    'cocula': ('Cocula', 'Jalisco'),
    'salto': ('Salto', 'Jalisco'),
    'guadalajara': ('Guadalajara', 'Jalisco'),
    'jalostitlán': ('Jalostitlán', 'Jalisco'),
    'jocotepec': ('Jocotepec', 'Jalisco'),
    'la barca': ('La Barca', 'Jalisco'),
    'lagos de moreno': ('Lagos De Moreno', 'Jalisco'),
    'magdalena': ('Magdalena', 'Sonora'),
    'tlaquepaque': ('Tlaquepaque', 'Jalisco'),
    'zapopan': ('Zapopan', 'Jalisco'),
    'zapotlán el grande': ('Zapotlán El Grande', 'Jalisco'),
    'zapotlanejo': ('Zapotlanejo', 'Jalisco'),
    'ocotlán': ('Ocotlán', 'Jalisco'),
    'puerto vallarta': ('Puerto Vallarta', 'Jalisco'),
    'san juan de los lagos': ('San Juan De Los Lagos', 'Jalisco'),
    'san miguel el alto': ('San Miguel El Alto', 'Jalisco'),
    'tala': ('Tala', 'Jalisco'),
    'tlajomulco de zúñiga': ('Tlajomulco De Zúñiga', 'Jalisco'),
    'acapulco de juárez': ('Acapulco De Juárez', 'Guerrero'),
    'chilpancingo de los bravo': ('Chilpancingo De Los Bravo', 'Guerrero'),
    'iguala de la independencia': ('Iguala De La Independencia', 'Guerrero'),
    'taxco de alarcón': ('Taxco De Alarcón', 'Guerrero'),
    'zihuatanejo (ixtapa)': ('Zihuatanejo (Ixtapa)', 'Guerrero'),
    'acaponeta': ('Acaponeta', 'Nayarit'),
    'bahía de banderas': ('Bahía De Banderas', 'Nayarit'),
    'compostela': ('Compostela', 'Nayarit'),
    'rosa morada': ('Rosa Morada', 'Nayarit'),
    'ruiz': ('Ruiz', 'Nayarit'),
    'santiago ixcuintla': ('Santiago Ixcuintla', 'Nayarit'),
    'santa maría del oro': ('Santa María Del Oro', 'Nayarit'),
    'tecuala': ('Tecuala', 'Nayarit'),
    'tepic': ('Tepic', 'Nayarit'),
    'tuxpan': ('Tuxpan', 'Nayarit'),
    'xalisco': ('Xalisco', 'Nayarit'),
    'actopan': ('Actopan', 'Hidalgo'),
    'apan': ('Apan', 'Hidalgo'),
    'atitalaquia': ('Atitalaquia', 'Hidalgo'),
    'atotonilco el grande': ('Atotonilco El Grande', 'Hidalgo'),
    'cuautepec de hinojosa': ('Cuautepec De Hinojosa', 'Hidalgo'),
    'huichapan': ('Huichapan', 'Hidalgo'),
    'ixmiquilpan': ('Ixmiquilpan', 'Hidalgo'),
    'mineral de la reforma': ('Mineral De La Reforma', 'Hidalgo'),
    'mixquiahuala de juárez': ('Mixquiahuala De Juárez', 'Hidalgo'),
    'pachuca de soto': ('Pachuca De Soto', 'Hidalgo'),
    'santiago tulantepec de lugo': ('Santiago Tulantepec De Lugo', 'Hidalgo'),
    'guerrero': ('Guerrero', 'Hidalgo'),
    'tepeapulco': ('Tepeapulco', 'Hidalgo'),
    'tepeji del río ocampo': ('Tepeji Del Río Ocampo', 'Hidalgo'),
    'tizayuca': ('Tizayuca', 'Hidalgo'),
    'tlahuelilpan': ('Tlahuelilpan', 'Hidalgo'),
    'tula de allende': ('Tula De Allende', 'Hidalgo'),
    'tulancingo de bravo': ('Tulancingo De Bravo', 'Hidalgo'),
    'juitepec': ('Juitepec', 'Hidalgo'),
    'yautepec': ('Yautepec', 'Hidalgo'),
    'apatzingán de la construcción': ('Apatzingán De La Construcción', 'Michoacán'),
    'cd hidalgo': ('Cd Hidalgo', 'Michoacán'),
    'jacona': ('Jacona', 'Michoacán'),
    'jiquilpan': ('Jiquilpan', 'Michoacán'),
    'la piedad de cabadas': ('La Piedad De Cabadas', 'Michoacán'),
    'lázaro cárdenas': ('Lázaro Cárdenas', 'Michoacán'),
    'maravatío de ocampo': ('Maravatío De Ocampo', 'Michoacán'),
    'morelia': ('Morelia', 'Michoacán'),
    'pátzcuaro': ('Pátzcuaro', 'Michoacán'),
    'purépero ichais': ('Purépero Ichais', 'Michoacán'),
    'quiroga': ('Quiroga', 'Michoacán'),
    'sahuayo de morelos': ('Sahuayo De Morelos', 'Michoacán'),
    'tarímbaro': ('Tarímbaro', 'Michoacán'),
    'uruapan': ('Uruapan', 'Michoacán'),
    'zacapu': ('Zacapu', 'Michoacán'),
    'zamora de hidalgo': ('Zamora De Hidalgo', 'Michoacán'),
    'acajete': ('Acajete', 'Puebla'),
    'acatlán': ('Acatlán', 'Puebla'),
    'acatzingo': ('Acatzingo', 'Puebla'),
    'amozoc': ('Amozoc', 'Puebla'),
    'atilizco': ('Atilizco', 'Puebla'),
    'chignahuapan': ('Chignahuapan', 'Puebla'),
    'chinautla': ('Chinautla', 'Puebla'),
    'coapiaxcla de madero': ('Coapiaxcla De Madero', 'Puebla'),
    'cuautlancingo': ('Cuautlancingo', 'Puebla'),
    'izúcar de matamoros': ('Izúcar De Matamoros', 'Puebla'),
    'libres': ('Libres', 'Puebla'),
    'puebla': ('Puebla', 'Puebla'),
    'quecholac': ('Quecholac', 'Puebla'),
    'san jose chiapa': ('San Jose Chiapa', 'Puebla'),
    'san martín texmelucan': ('San Martín Texmelucan', 'Puebla'),
    'san pedro cholula': ('San Pedro Cholula', 'Puebla'),
    'san salvador huixcolotla': ('San Salvador Huixcolotla', 'Puebla'),
    'san salvador huixcolotia': ('San Salvador Huixcolotia', 'Puebla'),
    'tehuacán': ('Tehuacán', 'Puebla'),
    'tepeaca': ('Tepeaca', 'Puebla'),
    'teziutlán': ('Teziutlán', 'Puebla'),
    'tecamachalco': ('Tecamachalco', 'Puebla'),
    'zicotepec': ('Zicotepec', 'Puebla'),
    'zacatlán': ('Zacatlán', 'Puebla'),
    'agua prieta': ('Agua Prieta', 'Sonora'),
    'caborca': ('Caborca', 'Sonora'),
    'cd obregón': ('Cd Obregón', 'Sonora'),
    'cananea': ('Cananea', 'Sonora'),
    'etchojoa': ('Etchojoa', 'Sonora'),
    'guaymas': ('Guaymas', 'Sonora'),
    'hermosillo': ('Hermosillo', 'Sonora'),
    'huatabampo': ('Huatabampo', 'Sonora'),
    'imuris': ('Imuris', 'Sonora'),
    'navojoa': ('Navojoa', 'Sonora'),
    'nogales': ('Nogales', 'Sonora'),
    'pto peñasco': ('Pto Peñasco', 'Sonora'),
    'sta ana': ('Sta Ana', 'Sonora'),
    'asunción nochixtlán': ('Asunción Nochixtlán', 'Oaxaca'),
    'huajuapan de león': ('Huajuapan De León', 'Oaxaca'),
    'oaxaca de juárez': ('Oaxaca De Juárez', 'Oaxaca'),
    'san agustín de las juntas': ('San Agustín De Las Juntas', 'Oaxaca'),
    'san agustín etla': ('San Agustín Etla', 'Oaxaca'),
    'san andrés huayápam': ('San Andrés Huayápam', 'Oaxaca'),
    'san antonio de la cal': ('San Antonio De La Cal', 'Oaxaca'),
    'san jacinto amilpas': ('San Jacinto Amilpas', 'Oaxaca'),
    'san juan bautista tuxtepec': ('San Juan Bautista Tuxtepec', 'Oaxaca'),
    'san lorenzo cacaotepec': ('San Lorenzo Cacaotepec', 'Oaxaca'),
    'san pablo etla': ('San Pablo Etla', 'Oaxaca'),
    'san sebastián tutla': ('San Sebastián Tutla', 'Oaxaca'),
    'santa cruz amilpas': ('Santa Cruz Amilpas', 'Oaxaca'),
    'santa cruz xoxocotlán': ('Santa Cruz Xoxocotlán', 'Oaxaca'),
    'santa lucía del camino': ('Santa Lucía Del Camino', 'Oaxaca'),
    'santa maría del tule': ('Santa María Del Tule', 'Oaxaca'),
    'santiago matatlán': ('Santiago Matatlán', 'Oaxaca'),
    'tlalixtla': ('Tlalixtla', 'Oaxaca'),
    'villa de etla': ('Villa De Etla', 'Oaxaca'),
}

DESTINOS_OCURRE = {
    'aguascalientes': ('Aguascalientes', 'Aguascalientes'),
    'ensenada': ('Ensenada', 'Baja California'),
    'mexicali': ('Mexicali', 'Baja California'),
    'tijuana': ('Tijuana', 'Baja California'),
    'la paz': ('La Paz', 'Baja California Sur'),
    'los cabos': ('Los Cabos', 'Baja California Sur'),
    'campeche': ('Campeche', 'Campeche'),
    'atlacomulco': ('Atlacomulco', 'Ciudad de México'),
    'av. central': ('Av. Central', 'Ciudad de México'),
    'av.revolucion': ('Av.Revolucion', 'Ciudad de México'),
    'cuautitlán': ('Cuautitlán', 'Ciudad de México'),
    'ecatepec 2': ('Ecatepec 2', 'Ciudad de México'),
    'gustavo baz': ('Gustavo Baz', 'Ciudad de México'),
    'huehuehuetoca': ('Huehuehuetoca', 'Ciudad de México'),
    'iztacalco': ('Iztacalco', 'Ciudad de México'),
    'ignacio zaragoza': ('Ignacio Zaragoza', 'Ciudad de México'),
    'marina nacional': ('Marina Nacional', 'Ciudad de México'),
    'naucalpan': ('Naucalpan', 'Ciudad de México'),
    'tepotzotlán': ('Tepotzotlán', 'Ciudad de México'),
    'texcoco': ('Texcoco', 'Ciudad de México'),
    'tlalnepantla': ('Tlalnepantla', 'Ciudad de México'),
    'toluca': ('Toluca', 'Ciudad de México'),
    'tultitlán': ('Tultitlán', 'Ciudad de México'),
    'vallejo': ('Vallejo', 'Ciudad de México'),
    'vallejo 2': ('Vallejo 2', 'Ciudad de México'),
    'san cristóbal de las casas': ('San Cristóbal De Las Casas', 'Chiapas'),
    'tapachula': ('Tapachula', 'Chiapas'),
    'tuxtla gtz': ('Tuxtla Gtz', 'Chiapas'),
    'cd juárez': ('Cd Juárez', 'Chihuahua'),
    'chihuahua': ('Chihuahua', 'Chihuahua'),
    'ramos arizpe': ('Ramos Arizpe', 'Coahuila'),
    'saltillo': ('Saltillo', 'Coahuila'),
    'torreón': ('Torreón', 'Coahuila'),
    'colima': ('Colima', 'Colima'),
    'manzanillo': ('Manzanillo', 'Colima'),
    'durango': ('Durango', 'Durango'),
    'gómez palacio': ('Gómez Palacio', 'Durango'),
    'celaya': ('Celaya', 'Guanajuato'),
    'dolores hidalgo': ('Dolores Hidalgo', 'Guanajuato'),
    'irapuato': ('Irapuato', 'Guanajuato'),
    'león aeropuerto': ('León Aeropuerto', 'Guanajuato'),
    'león central camionera': ('León Central Camionera', 'Guanajuato'),
    'moroleón': ('Moroleón', 'Guanajuato'),
    'purísima del rincón': ('Purísima Del Rincón', 'Guanajuato'),
    'san miguel de allende': ('San Miguel De Allende', 'Guanajuato'),
    'acapulco': ('Acapulco', 'Guerrero'),
    'pachuca': ('Pachuca', 'Hidalgo'),
    'arandas': ('Arandas', 'Jalisco'),
    'guadalajara': ('Guadalajara', 'Jalisco'),
    'lagos de moreno': ('Lagos De Moreno', 'Jalisco'),
    'ocotlán': ('Ocotlán', 'Jalisco'),
    'puerto vallarta': ('Puerto Vallarta', 'Jalisco'),
    'san juan de los lagos': ('San Juan De Los Lagos', 'Jalisco'),
    'san miguel el alto': ('San Miguel El Alto', 'Jalisco'),
    'tepatitlán': ('Tepatitlán', 'Jalisco'),
    'zapopan periférico': ('Zapopan Periférico', 'Jalisco'),
    'zapotlanejo': ('Zapotlanejo', 'Jalisco'),
    'la piedad': ('La Piedad', 'Michoacán'),
    'lázaro cárdenas': ('Lázaro Cárdenas', 'Michoacán'),
    'morelia': ('Morelia', 'Michoacán'),
    'morelia tarimbaro': ('Morelia Tarimbaro', 'Michoacán'),
    'sahuayo': ('Sahuayo', 'Michoacán'),
    'uruapan': ('Uruapan', 'Michoacán'),
    'zamora': ('Zamora', 'Michoacán'),
    'cuernavaca': ('Cuernavaca', 'Morelos'),
    'tepic': ('Tepic', 'Nayarit'),
    'apodaca': ('Apodaca', 'Nuevo León'),
    'escobedo': ('Escobedo', 'Nuevo León'),
    'guadalupe': ('Guadalupe', 'Nuevo León'),
    'lincoln 2': ('Lincoln 2', 'Nuevo León'),
    'monterrey estadio': ('Monterrey Estadio', 'Nuevo León'),
    'monterrey universidad': ('Monterrey Universidad', 'Nuevo León'),
    'santa catarina paseo': ('Santa Catarina Paseo', 'Nuevo León'),
    'santa catarina cedis': ('Santa Catarina Cedis', 'Nuevo León'),
    'mazatlán': ('Mazatlán', 'Sinaloa'),
    'oaxaca': ('Oaxaca', 'Oaxaca'),
    'puebla av.31': ('Puebla Av.31', 'Puebla'),
    'puebla estadio': ('Puebla Estadio', 'Puebla'),
    'corregidora': ('Corregidora', 'Querétaro'),
    'el marquez': ('El Marquez', 'Querétaro'),
    'querétaro': ('Querétaro', 'Querétaro'),
    'san juan del río': ('San Juan Del Río', 'Querétaro'),
    'cancún': ('Cancún', 'Quintana Roo'),
    'san luis potosí': ('San Luis Potosí', 'San Luis Potosí'),
    'soledad de graciano': ('Soledad De Graciano', 'San Luis Potosí'),
    'culiacán': ('Culiacán', 'Sinaloa'),
    'culiacán pedro infante': ('Culiacán Pedro Infante', 'Sinaloa'),
    'los mochis': ('Los Mochis', 'Sinaloa'),
    'hermosillo cedis': ('Hermosillo Cedis', 'Sonora'),
    'nogales': ('Nogales', 'Sonora'),
    'cd obregón': ('Cd Obregón', 'Sonora'),
    'tlaxcala': ('Tlaxcala', 'Tlaxcala'),
    'villahermosa': ('Villahermosa', 'Tabasco'),
    'nuevo laredo': ('Nuevo Laredo', 'Tamaulipas'),
    'reynosa': ('Reynosa', 'Tamaulipas'),
    'tampico': ('Tampico', 'Tamaulipas'),
    'coatzacoalcos': ('Coatzacoalcos', 'Veracruz'),
    'veracruz': ('Veracruz', 'Veracruz'),
    'xalapa': ('Xalapa', 'Veracruz'),
    'mérida': ('Mérida', 'Yucatán'),
    'ticul': ('Ticul', 'Yucatán'),
    'zacatecas': ('Zacatecas', 'Zacatecas'),
}

def _check_destino_especial(texto: str):
    """
    Busca en el texto si se menciona alguna ciudad de los listados especiales.
    Devuelve ('domicilio'|'ocurre'|'ambos', nombre_ciudad, estado) o None.
    """
    texto_norm = _norm_ciudad(texto)

    # Construir índice normalizado sin acentos → (tipo, nombre, estado, key_original)
    # Se hace en cada llamada pero puede cachearse si se vuelve necesario
    index = {}  # norm_key -> (tipo, nombre, estado)
    all_keys = sorted(
        set(list(DESTINOS_DOMICILIO.keys()) + list(DESTINOS_OCURRE.keys())),
        key=len, reverse=True  # Más largos primero para preferir "playa del carmen" > "carmen"
    )
    for k in all_keys:
        k_norm = _norm_ciudad(k)
        in_dom = k in DESTINOS_DOMICILIO
        in_ocu = k in DESTINOS_OCURRE
        nombre, estado = (DESTINOS_DOMICILIO if in_dom else DESTINOS_OCURRE)[k]
        if k_norm not in index:  # No sobreescribir si ya existe una entrada más larga
            tipo = 'ambos' if (in_dom and in_ocu) else ('domicilio' if in_dom else 'ocurre')
            index[k_norm] = (tipo, nombre, estado)

    # Buscar ciudades en el texto normalizado
    for k_norm in sorted(index.keys(), key=len, reverse=True):
        if k_norm in texto_norm:
            return index[k_norm]
    return None

# === MEDIDAS_LIST (etiquetas reales que manejas) ===
MEDIDAS_LIST = [
    # Galones
    "18gal","20gal","22gal","27gal","30gal","35gal","40gal","45gal","50gal","55gal","57gal","60gal","65gal","70gal","75gal","77gal",

    # Home Depot / Uhaul
    "HD small","HD Med-Uhaul Mediana","HD Large","HD XL",
    "Uhaul 24*24*20","Uhaul 24.5*24.5*27.5",

    # Cajas
    "Caja 18*18*24","Caja 22*22*22","Caja 24*24*24",
    "Caja 24*24*26","Caja 24*24*44","Caja 24*24*48",

    # Bicis / mochilas / TVs
    "Bici niño","Bici med","Bici adulto",
    "Mochila de escuela","Mochila 5 llantas","Mochila 6 llantas",
    "TV 30-39","TV 40-49","TV 50-55","TV 60-65","TV 70-75","TV 80-85",
]


def _get_usa_ruta_contacto(texto: str):
    """
    Busca si en el texto se menciona alguna ciudad/estado de las rutas de USA.
    Devuelve el teléfono correspondiente y el nombre encontrado, o (None, None).
    """
    texto_norm = _norm_ciudad(texto)
    # Buscar coincidencias en el texto normalizado
    # Se recorre el diccionario para encontrar la clave que esté presente en el texto
    for key, phone in _USA_RUTAS_MAP.items():
        if key in texto_norm:
            return phone, key
    return None, None


# ======================
# FUNCIÓN PRINCIPAL
# ======================

def chatbot_response(user_text: str) -> dict:
    """
    Respuesta controlada:
    - No inventa: devuelve SOLO plantillas (responses) o mensajes fijos.
    - Devuelve: tag, confidence, response
    """
    user_text = (user_text or "").strip()

    # 0) Vacío
    if not user_text:
        return {"tag": "error", "confidence": 0.0, "response": "Por favor escribe un mensaje."}

    # Normalización básica para Early Checks
    text_n = _norm_ciudad(user_text)

    # 0.0a) Clausulas
    import re
    clausulas_re = re.compile(r"\b(clausulas|cláusulas|reglas|politicas|políticas|condiciones|terminos|términos)\b", re.IGNORECASE)
    if clausulas_re.search(text_n):
        return {"tag": "clausulas", "confidence": 1.0, "response": "Aquí te comparto nuestras cláusulas y condiciones de servicio. Para mayor detalle, te invitamos a consultar el desglose total con tu asesor en línea. 📄👇"}

    # 0.0b) Restricciones especiales: NO USA destination, NO MEX origin
    to_usa_re = re.compile(r"\b(a|para|hacia|enviar\s+a|mandar\s+a)\s+(usa|eeuu|estados\s+unidos|texas|california|tennessee|georgia|alabama|carolina|florida)\b", re.IGNORECASE)
    from_mex_re = re.compile(r"\b(de|desde|saliendo\s+de)\s+(mexico|méxico|guanajuato|queretaro|querétaro|puebla|michoacan|jalisco)\b", re.IGNORECASE)
    
    if to_usa_re.search(text_n):
        return {"tag": "restriction_to_usa", "confidence": 1.0, "response": "⚠️ Lo sentimos, nuestro servicio de paquetería es **exclusivo de Estados Unidos hacia México**. No realizamos envíos dentro de Estados Unidos ni de México hacia Estados Unidos. ¡Agradecemos tu comprensión! 😊"}
    if from_mex_re.search(text_n):
        return {"tag": "restriction_from_mexico", "confidence": 1.0, "response": "⚠️ Lo sentimos, solo realizamos envíos con origen en **Estados Unidos** (Georgia, Alabama, Texas, Tennessee) con destino hacia **México**, pero **no de México hacia Estados Unidos**. ¡Agradecemos tu comprensión! 😊"}

    # 0.0c) Saludos
    if SALUDO_RE.search(text_n):
        return {"tag": "saludo", "confidence": 1.0, "response": _pick_response("saludo")}

    # 0.0a) Despedidas / agradecimientos
    if DESPEDIDA_RE.search(text_n):
        return {"tag": "despedida", "confidence": 1.0, "response": _pick_response("despedida")}

    # 1) Identidad del Bot (¿Quién eres?, ¿Eres un robot?)
    if IDENTIDAD_BOT_RE.search(text_n):
        return {"tag": "identidad_superpacky", "confidence": 1.0, "response": _pick_response("identidad_superpacky")}

    # 2) Artículos Prohibidos (Oro, Plata, Armas, Animales, etc.)
    if ARTICULOS_PROHIBIDOS_RE.search(text_n):
        return {"tag": "que_no_puedo_enviar", "confidence": 1.0, "response": _pick_response("que_no_puedo_enviar")}

    # 3) Fechas de salida / Viajes (PRIORIDAD sobre cotización)
    if SALIDA_RE.search(text_n):
        return {"tag": "salidas_info", "confidence": 1.0, "response": _SALIDA_RESPONSE}

    # 4) Rastreo Estricto (Guía, Tracking)
    if RASTREO_ESTRICTO_RE.search(text_n):
        return {"tag": "rastreo_paquete", "confidence": 1.0, "response": _pick_response("rastreo_paquete")}

    # 5) Menciones de rutas específicas en USA (Dalton, Austin, etc.)
    usa_phone, city_found = _get_usa_ruta_contacto(user_text)
    if usa_phone:
        return {
            "tag": "contacto_ruta_usa",
            "confidence": 1.0,
            "response": (
                f"¡Perfecto! 😊 Para darte mayores informes sobre recolección en o cerca de {city_found.title()}, "
                f"favor de comunicarte directamente al:\n\n"
                f"📲 {usa_phone}\n\n"
                f"Ellos se encargan de esa ruta y podrán confirmarte si el transportista puede pasar por tu paquete y coordinar los detalles con gusto. 🙌"
            )
        }

    # 6) Envíos desde otro estado (Origen no GA/TX/TN/AL)
    if ORIGEN_OTRO_RE.search(user_text):
        return {
            "tag": "origen_no_valido",
            "confidence": 1.0,
            "response": _ORIGEN_OTRO_RESPONSE
        }

    # 0.0b) Consulta de ciudades por estado — detectado ANTES de restriccion_geografica
    # Ej: "qué ciudades de Tabasco visitan", "qué ciudades tienen en Michoacán"
    _COB_PRECOZ_RE = re.compile(
        r"(qu[eé]\s+ciudades|qu[eé]\s+lugares|d[oó]nde\s+llegan|d[oó]nde\s+entregan|qu[eé]\s+destinos).{0,30}\b("
        r"aguascalientes|baja\s+california\s+sur|baja\s+california|campeche|chiapas|chihuahua|"
        r"ciudad\s+de\s+m[eé]xico|cdmx|coahuila|colima|durango|guanajuato|guerrero|hidalgo|"
        r"jalisco|michoac[aá]n|morelos|nayarit|nuevo\s+le[oó]n|oaxaca|puebla|quer[eé]taro|"
        r"quintana\s+roo|san\s+luis\s+potos[ií]|sinaloa|sonora|tabasco|tamaulipas|"
        r"tlaxcala|veracruz|yucat[aá]n|zacatecas|estado\s+de\s+m[eé]xico|edomex)\b",
        re.IGNORECASE
    )
    # También: "ciudades de [estado] visitan", "[estado] qué ciudades"
    _COB_PRECOZ_INV_RE = re.compile(
        r"\b(aguascalientes|baja\s+california\s+sur|baja\s+california|campeche|chiapas|chihuahua|"
        r"ciudad\s+de\s+m[eé]xico|cdmx|coahuila|colima|durango|guanajuato|guerrero|hidalgo|"
        r"jalisco|michoac[aá]n|morelos|nayarit|nuevo\s+le[oó]n|oaxaca|puebla|quer[eé]taro|"
        r"quintana\s+roo|san\s+luis\s+potos[ií]|sinaloa|sonora|tabasco|tamaulipas|"
        r"tlaxcala|veracruz|yucat[aá]n|zacatecas|estado\s+de\s+m[eé]xico|edomex)\b.{0,30}"
        r"(qu[eé]\s+ciudades|qu[eé]\s+lugares|visitan|atienden|llegan|tienen)",
        re.IGNORECASE
    )
    _cob_precoz = _COB_PRECOZ_RE.search(user_text) or _COB_PRECOZ_INV_RE.search(user_text)
    if _cob_precoz:
        _text_norm_p = _norm_est(user_text)
        _est_p = None
        for _k, _v in ESTADO_NORM_MAP.items():
            if _k and len(_k) >= 5 and _k in _text_norm_p:
                _est_p = _v
                break
        if _est_p:
            _ciudades_p = CIUDADES_POR_ESTADO.get(_est_p, [])
            if _ciudades_p:
                _ciudades_str = "\n".join(f"• {c}" for c in _ciudades_p)
                return {
                    "tag": "cobertura_estado",
                    "confidence": 1.0,
                    "response": (
                        f"¡Claro! Realizamos envíos al estado de {_est_p}. 😊\n\n"
                        f"Tenemos cobertura en las siguientes ciudades:\n\n{_ciudades_str}\n\n"
                        f"¿Desde qué estado de EE.UU. piensas enviar? 📦"
                    )
                }

    # 0.1) Restricción: Solo EE.UU. a México
    # Si detectamos un origen que parece ser México, bloqueamos.
    # Pero NO si es una pregunta genérica de cobertura/destino
    if MEXICO_ORIGIN_RE.search(user_text) and not DESTINATION_Q_RE.search(user_text):
        return {
            "tag": "restriccion_geografica",
            "confidence": 1.0,
            "response": _pick_response("restriccion_geografica")
        }

    # 0.1a.0) Solicitud de contacto / teléfono / asesor humano
    if CONTACTO_RE.search(user_text):
        return {
            "tag": "contacto_mexico",
            "confidence": 1.0,
            "response": _pick_response("contacto_mexico")
        }

    # ── Bloque de Early-Checks P1 ────────────────────────────────────────
    # 0.1b) Puntos de entrega en EE.UU. (los 4 estados de origen + 2 puntos físicos)
    if PUNTOS_USA_RE.search(user_text):
        return {
            "tag": "ubicaciones_usa",
            "confidence": 1.0,
            "response": _PUNTOS_USA_RESPONSE
        }

    # 0.1c) Rastreo / tracking
    if RASTREO_RE.search(user_text):
        return {
            "tag": "rastreo",
            "confidence": 1.0,
            "response": _pick_response("rastreo")
        }

    # 0.1d) Tiempos de entrega
    if TIEMPO_RE.search(user_text):
        return {
            "tag": "tiempos_entrega",
            "confidence": 1.0,
            "response": _pick_response("tiempos_entrega")
        }

    # 1.1) Historia / Antigüedad — EARLY
    if FUNDACION_RE.search(user_text):
        return {
            "tag": "empresa_fundacion",
            "confidence": 1.0,
            "response": _pick_response("empresa_fundacion")
        }

    # 0.1e) Artículos PROHIBIDOS (va antes de permitidos — más específico)
    if ARTICULOS_PROHIBIDOS_RE.search(user_text):
        return {
            "tag": "que_no_puedo_enviar",
            "confidence": 1.0,
            "response": _pick_response("que_no_puedo_enviar")
        }

    # 0.1f) Artículos permitidos / qué se puede enviar
    if ARTICULOS_PERMITIDOS_RE.search(user_text):
        return {
            "tag": "que_puedo_enviar",
            "confidence": 1.0,
            "response": _pick_response("que_puedo_enviar")
        }

    # 0.1g) Aduana / revisión de paquetes
    if ADUANA_RE.search(user_text):
        return {
            "tag": "revisan_paquetes",
            "confidence": 1.0,
            "response": _pick_response("revisan_paquetes")
        }

    # 0.1h) Cambio de dirección de entrega
    if CAMBIO_DIR_RE.search(user_text):
        return {
            "tag": "cambiar_direccion_entrega",
            "confidence": 1.0,
            "response": _pick_response("cambiar_direccion_entrega")
        }

    # 0.1i) Cotización / precio — DESPUÉS de rastreo y tiempos para evitar colisiones
    if COTIZAR_RE.search(user_text):
        return {
            "tag": "cotizacion",
            "confidence": 1.0,
            "response": _pick_response("cotizacion")
        }

    # 0.1j) Fechas de salida / Próximos viajes
    if SALIDA_RE.search(user_text):
        return {
            "tag": "salidas_info",
            "confidence": 1.0,
            "response": _SALIDA_RESPONSE
        }

    # 0.1k) Envíos desde otro estado (Origen no GA/TX/TN/AL)
    if ORIGEN_OTRO_RE.search(user_text):
        return {
            "tag": "origen_no_valido",
            "confidence": 1.0,
            "response": _ORIGEN_OTRO_RESPONSE
        }

    # 0.1l) "Quiero hacer un envío a [Ciudad]" -> Valida cobertura directamente
    match_envio = ENVIO_A_CIUDAD_RE.search(user_text)
    if match_envio:
        ciudad_mencionada = match_envio.group(1).strip()
        # Intentar validar la ciudad capturada
        res_ciudad = _check_destino_especial(ciudad_mencionada)
        if res_ciudad:
            tipo, nombre_c, estado_c = res_ciudad
            return {
                "tag": "confirmacion_destino",
                "confidence": 1.0,
                "response": (
                    f"¡Claro que sí! 😊 Hacemos envíos a {nombre_c}, {estado_c}.\n\n"
                    "Para darte una cotización exacta, solo dime:\n"
                    "• 📦 Medidas del paquete (largo × ancho × alto)\n\n"
                    "¿Qué medidas tiene tu caja o barril? 🙌"
                )
            }
        else:
            # Verificar si lo capturado es un estado (ej: "Michoacán") en vez de una ciudad
            _est_capturado = ESTADO_NORM_MAP.get(_norm_est(ciudad_mencionada))
            if not _est_capturado:
                # Buscar también si el nombre del estado está contenido en lo capturado
                _text_cap_norm = _norm_est(ciudad_mencionada)
                for _k, _v in ESTADO_NORM_MAP.items():
                    if _k and len(_k) >= 5 and _k in _text_cap_norm:
                        _est_capturado = _v
                        break

            if _est_capturado:
                _ciudades_est = CIUDADES_POR_ESTADO.get(_est_capturado, [])
                if _ciudades_est:
                    _ciudades_str = "\n".join(f"• {c}" for c in _ciudades_est)
                    return {
                        "tag": "cobertura_estado",
                        "confidence": 1.0,
                        "response": (
                            f"¡Claro! Realizamos envíos al estado de {_est_capturado}. 😊\n\n"
                            f"Tenemos cobertura en las siguientes ciudades:\n\n{_ciudades_str}\n\n"
                            f"¿Desde qué estado de EE.UU. piensas enviar? 📦"
                        )
                    }

            # Si no se reconoce como ciudad ni como estado, sugerir contacto
            return {
                "tag": "duda_destino_contacto",
                "confidence": 0.9,
                "response": (
                    f"He notado que quieres hacer un envío a '{ciudad_mencionada}'. "
                    "Para confirmarte cobertura exacta en ese punto, favor de comunicarte al:\n\n"
                    "📲 +52 418 110 7243\n\n"
                    "Diles a que ciudad quieres mandar y con gusto te informarán. 😊"
                )
            }

    # ── Fin bloque Early-Checks P1 ───────────────────────────────────────

    # 0.1a.COBX) Consulta de estado específico
    # Detecta si el texto es un nombre de estado o contiene uno
    _est_match = None
    _user_norm = _norm_est(user_text.strip())
    # 1) Texto exacto = nombre de estado
    if _user_norm in ESTADO_NORM_MAP:
        _est_match = ESTADO_NORM_MAP[_user_norm]
    else:
        # 2) El texto contiene el nombre de algún estado (búsqueda directa normalizada)
        _text_norm_full = _norm_est(user_text)
        for _k, _v in ESTADO_NORM_MAP.items():
            if _k and len(_k) >= 5 and _k in _text_norm_full:
                _est_match = _v
                break

    if _est_match and not MEXICO_ORIGIN_RE.search(user_text):
        ciudades = CIUDADES_POR_ESTADO.get(_est_match, [])
        if ciudades:
            _ciudades_str = "\n".join(f"• {c}" for c in ciudades)
            return {
                "tag": "cobertura_estado",
                "confidence": 1.0,
                "response": (
                    f"¡Claro! Realizamos envíos al estado de {_est_match}. 😊\n\n"
                    f"Tenemos cobertura en las siguientes ciudades:\n\n{_ciudades_str}\n\n"
                    f"¿Desde qué estado de EE.UU. piensas enviar? 📦"
                )
            }

    # 0.1a.COBG) Pregunta general de cobertura → mostrar lista de estados
    if COBERTURA_GENERAL_RE.search(user_text) and not ESTADO_CONSULTA_RE.search(user_text):
        estados_str = "\n".join(f"• {e}" for e in ESTADOS_LIST)
        return {
            "tag": "cobertura_general",
            "confidence": 1.0,
            "response": (
                "¡Sí! Cubrimos envíos a toda la República Mexicana.\n\n"
                "Selecciona el estado al que quieres enviar para ver las ciudades con cobertura:\n\n"
                f"{estados_str}\n\n"
                "Solo escribe el nombre del estado. 😊"
            )
        }

    # 0.1a.1) Recolección / pickup — aclarar que NO es a domicilio, son puntos fijos
    # (va ANTES que PAGO_RE para evitar colisión con "cobran por la recolección")
    if RECOLECCION_RE.search(user_text):
        return {
            "tag": "puntos_eeuu",
            "confidence": 1.0,
            "response": (
                "No contamos con recolección a domicilio. 🏠❌\n\n"
                "Los paquetes deben llevarse a nuestros puntos de entrega fijos en EE.UU.:\n\n"
                "📍 Little Rock, Arkansas\n"
                "📍 Dalton, Georgia\n\n"
                "Si me dices en qué ciudad estás, te oriento al punto más cercano. 😊"
            )
        }

    # 0.1a.2) Métodos de pago — respuesta hardcodeada (tag no existe en intents.json)
    if PAGO_RE.search(user_text):
        return {
            "tag": "metodos_pago",
            "confidence": 1.0,
            "response": _PAGO_RESPONSE
        }

    # 0.1a.3) Quiere iniciar un pedido/envío → redirige a cotización
    if PEDIDO_RE.search(user_text):
        return {
            "tag": "cotizar",
            "confidence": 1.0,
            "response": _pick_response("cotizar")
        }

    # 0.1a.4) Incidencia post-envío (retraso, daño, paquete equivocado)
    if INCIDENCIA_RE.search(user_text):
        return {
            "tag": "incidencia_envio",
            "confidence": 1.0,
            "response": (
                "Lo sentimos mucho por el inconveniente. 😟\n\n"
                "Para atender este tipo de situaciones lo más rápido posible, "
                "te pedimos de favor que te comuniques directamente con nuestro equipo:\n\n"
                "📲 (706) 980 88 89\n"
                "📲 (737) 233 7008\n"
                "📲 (706) 260 1711\n\n"
                "Ellos podrán revisar el estatus de tu envío y darte resolución. 🙏"
            )
        }

    # 0.1a.5) Menciona origen válido (Georgia/Alabama/Texas/Tennessee) sin cotización → respuesta dual
    if ORIGEN_VALIDO_MENCIONA_RE.search(user_text) and not MEXICO_DEST_RE.search(user_text):
        return {
            "tag": "origen_valido_contacto",
            "confidence": 0.9,
            "response": (
                "¡Perfecto! 😊 Contamos con punto de recolección en tu área.\n\n"
                "Puedes contactarnos directamente para coordinar tu envío:\n"
                "📲 (706) 980 88 89\n"
                "📲 (737) 233 7008\n"
                "📲 (706) 260 1711\n\n"
                "O si prefieres una cotización rápida, dime:\n"
                "• 📦 Medidas del paquete (largo × ancho × alto)\n"
                "• 📍 Ciudad destino en México\n\n"
                "¿Cómo te podemos ayudar? 🙌"
            )
        }

    # 0.1a) Destino especial — Domicilio o Ocurre
    _dest_esp = _check_destino_especial(user_text)
    if _dest_esp and not MEXICO_ORIGIN_RE.search(user_text):
        tipo, ciudad, estado = _dest_esp
        if tipo == 'domicilio':
            resp = (
                f"¡Claro que sí! 😊 Para {ciudad} ({estado}) realizamos entregas "
                f"a domicilio. El costo varía según las medidas del paquete. "
                f"¿Te gustaría que te cotizáramos? 📦"
            )
        elif tipo == 'ocurre':
            resp = (
                f"¡Sí enviamos a {ciudad} ({estado})! 😊 Para esta ciudad las entregas "
                f"se realizan en nuestras oficinas de ocurre (el cliente pasa a recoger). "
                f"¿Te gustaría una cotización? 📦"
            )
        else:  # ambos
            resp = (
                f"¡Sí enviamos a {ciudad} ({estado})! 😊 Para esta ciudad contamos con "
                f"dos opciones de entrega:\n"
                f"• 🏠 A domicilio — el paquete llega directo a la puerta\n"
                f"• 🏢 Ocurre — el cliente lo recoge en nuestras oficinas\n\n"
                f"El costo varía según las medidas. ¿Te damos una cotización? 📦"
            )
        return {"tag": "destino_especial", "confidence": 1.0, "response": resp}

    # 0.1c) Destino México mencionado con preposición de envío
    # Ej: "quiero hacer un envío a Aguascalientes", "mando algo para Jalisco"
    # → El servicio SÍ cubre México como destino; hay que orientar sobre el origen
    m_dest = MEXICO_DEST_RE.search(user_text)
    if m_dest and not MEXICO_ORIGIN_RE.search(user_text):
        # Extraer el nombre del destino de forma legible
        dest_raw = m_dest.group(2) if m_dest.lastindex and m_dest.lastindex >= 2 else "ese destino"
        dest_nombre = dest_raw.strip().title()
        return {
            "tag": "destino_mexico_valido",
            "confidence": 1.0,
            "response": (
                f"¡Claro que sí! 😊 Enviamos a {dest_nombre} y a toda la República Mexicana. \n\n"
                f"El costo varía según las medidas de tu paquete y el estado de EE.UU. desde donde realizas el envío.\n\n"
                f"Contamos con recolección desde:\n"
                f"• 🟢 Georgia\n• 🟢 Alabama\n• 🟢 Texas\n• 🟢 Tennessee\n\n"
                f"¿Deseas que te demos una cotización o iniciar un pedido? 📦"
            )
        }

    # 0.1b) Estado de EE.UU. no válido como origen
    # Detecta frases como "envío de Arkansas", "si quisiera hacer un envío de Arkansas"
    if USA_INVALID_ORIGIN_RE.search(user_text) and not USA_VALID_ORIGIN_RE.search(user_text):
        return {
            "tag": "origen_invalido_eeuu",
            "confidence": 1.0,
            "response": (
                "¡Muchas gracias por tu interés en nuestros servicios! 😊 "
                "Lamentablemente, por el momento no contamos con recolección desde el estado que mencionas.\n\n"
                "Pero si gustas, puedes contactarnos al 📲 418 110 7243 para darte más informes y ver cómo podemos ayudarte.\n\n"
                "Actualmente operamos desde Georgia, Alabama, Texas y Tennessee. "
                "¿Puedo ayudarte si el envío sale desde alguno de estos estados?"
            )
        }

    # 0.2) Aclaración de peso
    if WEIGHT_RE.search(user_text) and not PRICE_Q_RE.search(user_text):
         return {
            "tag": "aclaracion_peso",
            "confidence": 1.0,
            "response": _pick_response("aclaracion_peso")
        }

    # 1) Prohibidos (bloqueo)
    item_prohibido = _contains_prohibited(user_text)
    
    # 1.1) ¿Es una pregunta de "puedo enviar [X]?"
    # Si el item NO es prohibido, respondemos que sí por defecto (permissiveness).
    puedo_enviar_re = re.compile(r"\b(puedo|se\s+puede|pueden)\s+(enviar|mandar|meter|llevar|trasladar)\s+(?:un|una|unos|unas|el|la|los|las|mi|mis)?\s*([^?]+)", re.IGNORECASE)
    match_envio = puedo_enviar_re.search(user_text)
    if match_envio:
        articulo_solicitado = match_envio.group(3).strip().lower()
        # Verificamos si lo que pidió está en la lista negra
        bloqueado = _contains_prohibited(articulo_solicitado)
        
        if not bloqueado:
            return {
                "tag": "que_puedo_enviar",
                "confidence": 1.0,
                "response": (
                    f"¡Claro que sí! ✅ Se puede enviar {articulo_solicitado} sin problemas, siempre que no sea un material peligroso, "
                    "inflamable o ilegal. ¿Deseas cotizar el envío o tienes otra duda?"
                )
            }

    if item_prohibido:
        return {
            "tag": "prohibido",
            "confidence": 1.0,
            "response": (
                f"Lo sentimos, por políticas de seguridad no podemos realizar envíos de {item_prohibido}. "
                "¿Deseas cotizar algún otro artículo permitido (ropa, electrónicos, herramientas)?"
            )
        }

    # 2) Identidad
    if IDENTITY_RE.search(user_text):
        return {"tag": "identidad_superpacky", "confidence": 1.0, "response": random.choice(IDENTITY_RESPONSES)}

    # 2.5) Medidas / tamaños (catálogo genérico)
    if SIZES_Q_RE.search(user_text):
        return {
            "tag": "medidas_paquetes",
            "confidence": 1.0,
            "response": (
                "Manejamos varias opciones de tamaño 📦:\n"
                "• Galones: 18gal, 20gal, 22gal, 27gal, 30gal... hasta 77gal\n"
                "• Cajas: 18*18*24, 22*22*22, 24*24*24, 24*24*26, 24*24*44, 24*24*48\n"
                "• Uhaul: 24*24*20, 24.5*24.5*27.5\n"
                "• TVs: 30-39\", 40-49\", 50-55\", 60-65\", 70-75\", 80-85\"\n"
                "• Bicis: niño, mediana, adulto\n"
                "• Mochilas: escuela, 5 llantas, 6 llantas\n\n"
                "¿Tienes las medidas exactas? Dime largo, ancho y alto (cm o pulgadas) y te sugiero la caja ideal."
            )
        }

    _MEDIDAS_AMPLIO_RE = re.compile(
        r"(\d+\s*[xX×]\s*\d+\s*[xX×]\s*\d+)"
        r"|(tv\s+(?:de\s+)?\d+\s*(?:pulgadas?|\"|\'))"
        r"|(televisi[oó]n\s+(?:de\s+)?\d+)"
        r"|(pantalla\s+(?:de\s+)?\d+)"
        r"|(bici(?:cleta)?\s+(?:de\s+)?(?:adulto|ni[ñn]o|mediana))"
        r"|(largo\s+\d+.{0,30}ancho\s+\d+)"
        r"|(caja\s+(?:de\s+)?\d+\s*(?:litros?|galones?|gal|cm|pulgadas?))",
        re.IGNORECASE
    )
    if (
        re.search(r"\b(medida|medidas|tama[ñn]|dimensi[oó]n|dimensiones|cu[aá]nto\s+mide|alto|ancho|largo)\b", user_text, re.IGNORECASE)
        or _MEDIDAS_AMPLIO_RE.search(user_text)
    ):
        dims, unit = parse_user_dimensions(user_text)
        if dims:
            # usa el catálogo REAL: las llaves que manejas (idealmente del backend)
            # Si aquí no tienes MEDIDAS_LIST, puedes reconstruirla desde intents o pegar una lista fija.
            # Lo mejor: copiar MEDIDAS_LIST al chatbot o importarla (te explico abajo).
            result = suggest_box(MEDIDAS_LIST, dims, unit)

            if result.get("unit") is None:
                cand = result.get("candidate_in")
                if cand:
                    return {"tag":"medidas_paquetes", "confidence":1.0, "response":(
                        f"Entendido ✅. Me diste {dims[0]}×{dims[1]}×{dims[2]} pero no veo unidad.\n"
                        f"Si son pulgadas, te conviene: {cand['key']}."
                    )}
                return {"tag":"medidas_paquetes", "confidence":1.0, "response":(
                    f"Entendido ✅. Me diste {dims[0]}×{dims[1]}×{dims[2]}, pero necesito la unidad (cm o pulgadas) "
                    "para sugerirte una caja exacta del catálogo."
                )}

            cand = result.get("candidate")
            if cand:
                return {"tag":"medidas_paquetes", "confidence":1.0, "response":(
                    f"Perfecto ✅. Con {dims[0]}×{dims[1]}×{dims[2]} {result['unit']}, "
                    f"la opción recomendada del catálogo es: {cand['key']}."
                )}
            return {"tag":"medidas_paquetes", "confidence":1.0, "response":(
                f"Con {dims[0]}×{dims[1]}×{dims[2]} {result['unit']} no encontré una caja exacta que la cubra en el catálogo.\n"
                "¿Quieres intentar con otra medida o decirme si se puede acomodar en otra orientación?"
            )}

        # si NO encontró números, cae a tu lógica de etiqueta exacta
        desc = describe_size(user_text)
        if desc:
            return {"tag":"medidas_paquetes", "confidence":1.0, "response": desc}

        return {"tag":"medidas_paquetes", "confidence":0.9, "response":(
            "Dime las medidas como largo 12 ancho 18 alto 20 (cm o pulgadas), o la etiqueta exacta "
            "(Caja 24*24*24, TV 50-55, 18gal) y te sugiero la mejor opción."
        )}

    # Cobertura/destino (ciudades a las que llegan)
    if DESTINATION_Q_RE.search(user_text):
        resp = _pick_response("cobertura")
        if not resp or resp in fallback_responses:
            resp = "Nuestro servicio tiene cobertura a todo México. El precio varía según el destino. ¿A qué ciudad necesitas el envío?"
        return {
            "tag": "cobertura",
            "confidence": 1.0,
            "response": resp
        }

    # Tiempo de entrega (prioritario sobre precio para "cuánto tarda")
    if TIEMPO_RE.search(user_text):
        resp = _pick_response("tiempo_entrega")
        if not resp or resp in fallback_responses:
            resp = "El tiempo de entrega varía según el destino ⏱️. En promedio los envíos tardan entre 5 y 15 días hábiles. ¿A qué ciudad es el envío para darte una estimación más precisa?"
        return {
            "tag": "tiempo_entrega",
            "confidence": 1.0,
            "response": resp
        }

    # Horarios (prioritario sobre el clasificador)
    if HORARIOS_RE.search(user_text):
        resp = _pick_response("horarios_atencion")
        if not resp or resp in fallback_responses:
            resp = "Nuestro horario de atención es de lunes a viernes 🕗 de 9:00 am a 6:00 pm y sábados de 9:00 am a 2:00 pm (hora del centro de México). Para más información llama al 418 110 7243."
        return {
            "tag": "horarios_atencion",
            "confidence": 1.0,
            "response": resp
        }

    if PRICE_Q_RE.search(user_text):
        return {
            "tag": "precio_info",
            "confidence": 1.0,
            "response": (
                "El precio depende del origen, el destino y el tamaño 📦.\n\n"
                "Para cotizar, si gustas, toca el botón Cotizar envío y lo hacemos paso a paso."
            )
        }

    if ROUTE_Q_RE.search(user_text):
        return {
            "tag": "rutas_envio",
            "confidence": 1.0,
            "response": _pick_response("rutas_envio")
        }

    if HELP_Q_RE.search(user_text):
        return {
            "tag": "ayuda_general",
            "confidence": 1.0,
            "response": _pick_response("ayuda_general")
        }

    if TRANSPORTE_RE.search(user_text):
        return {
            "tag": "transporte_envio",
            "confidence": 1.0,
            "response": _pick_response("transporte_envio")
        }

    # Ubicación / dónde dejar el paquete / sucursales
    if UBICACION_RE.search(user_text):
        resp = _pick_response("ubicacion_oficina_mexico")
        if not resp or resp in fallback_responses:
            resp = "Puedes entregar tu paquete en nuestra oficina en San Diego de la Unión, Gto. 📍 Mapa: https://maps.app.goo.gl/AeYsmgbVpHTZAYH59. Para más puntos de entrega comunícate al 418 110 7243."
        return {
            "tag": "ubicacion_oficina_mexico",
            "confidence": 1.0,
            "response": resp
        }

    # Servicio general / preguntas difusas encaminadas al negocio
    if SERVICIO_RE.search(user_text):
        return {
            "tag": "ayuda_general",
            "confidence": 1.0,
            "response": (
                "Con gusto te ayudo 😊. Somos Paquetería San Diego de la Unión, especializados en envíos de EE.UU. a México.\n"
                "Puedo orientarte sobre: precios, tiempos de entrega, qué puedes enviar, tamaños de caja y ubicaciones.\n"
                "¿Qué necesitas saber primero?"
            )
        }

    # Zelle — información sobre el servicio de pagos
    if ZELLE_RE.search(user_text):
        return {
            "tag": "zelle_info",
            "confidence": 1.0,
            "response": _pick_response("zelle_info")
        }

    # Pagos — métodos de pago en general
    if PAGOS_RE.search(user_text):
        return {
            "tag": "zelle_info",
            "confidence": 1.0,
            "response": (
                "Aceptamos Zelle para pagos desde EE.UU. 💳⚡ "
                "Es rápido, seguro y gratuito. También puedes preguntar a nuestro equipo sobre otras opciones. "
                "¿Deseas más información?"
            )
        }

    # Comandos del bot
    if COMANDOS_RE.search(user_text):
        return {
            "tag": "comandos_bot",
            "confidence": 1.0,
            "response": _pick_response("comandos_bot")
        }

    # Qué te puedo preguntar / funciones del bot
    if QUE_PREGUNTAR_RE.search(user_text):
        return {
            "tag": "que_preguntar",
            "confidence": 1.0,
            "response": _pick_response("que_preguntar")
        }

    # 3) Clasificación transformer supervisada (con heurística por keywords)
    top = _topk_probs(user_text, k=3)
    top = _keyword_boost(user_text, top)

    best_tag, best_p = top[0]
    second_p = top[1][1] if len(top) > 1 else 0.0

    # Umbral por intent (si aplica)
    min_p = PER_TAG_MIN_PROBA.get(best_tag, MIN_PROBA_TO_ACCEPT)

    # 4) Gate de aceptación: probabilidad y margen
    if best_p < min_p or (best_p - second_p) < MARGIN_TO_ACCEPT:
        # Si detecta intención de cotizar o sucursal, guía al flujo del sitio
        if QUOTE_HINT_RE.search(user_text):
            return {
                "tag": "cotizar",
                "confidence": float(best_p),
                "response": (
                    "Puedo ayudarte a cotizar. Dime por favor: origen, destino y medidas "
                    "(por ejemplo: 18gal, 22gal, Caja 24*24*24, TV 50-55)."
                )
            }
        if BRANCH_HINT_RE.search(user_text):
            # Si pregunta por oficinas/sucursales, damos la info de la oficina principal por defecto
            # si no detectamos una ciudad específica en los patterns de ubicación.
            return {
                "tag": "ubicacion_oficina_mexico",
                "confidence": float(best_p),
                "response": _pick_response("ubicacion_oficina_mexico")
            }

        return {"tag": "fallback", "confidence": float(best_p), "response": random.choice(fallback_responses)}

    # 5) Respuesta segura por plantillas
    return {"tag": best_tag, "confidence": float(best_p), "response": _pick_response(best_tag)}
