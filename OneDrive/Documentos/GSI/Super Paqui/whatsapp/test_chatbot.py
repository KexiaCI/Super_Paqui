import os
import chatbot as cb
from tabulate import tabulate

QUESTIONS = [
    # Intenciones de Cotizar / Enviar
    "Quiero enviar una caja", "Necesito cotizar un paquete", "Me podrían cotizar un envío",
    "¿Cuánto cobran por enviar ropa?", "¿Cuánto me sale mandar una televisión?",
    "Quiero mandar unas cosas a México", "Deseo realizar un envío", "Tengo unas cajas para exportar",
    "¿Cuál es el precio para enviar a Guanajuato?", "Necesito precios para mandar a Querétaro",
    "cotización", "cotizar", "quiero hacer un pedido", "hacer un envío", "quiero mandar un paquete",
    
    # Sucursales / Ubicaciones
    "¿Dónde están ubicados?", "¿Tienen sucursal en Texas?", "¿Me das la dirección de Atlanta?",
    "¿Dónde puedo ir a dejar mis cajas?", "dirección", "ubicación", "¿A dónde llevo mi paquete?",
    "busco una sucursal cerca de mí", "¿Tienen bodega en Alabama?", "pásame tu dirección",

    # Tiempos de Entrega
    "¿Cuánto tarda en llegar a México?", "¿Cuántos días se tardan en entregar?",
    "¿Si lo dejo hoy cuándo llega?", "tiempo de entrega", "¿En cuánto tiempo llega mi caja?",
    "¿Tardan mucho en entregar en Guanajuato?", "¿Cuándo llegaría mi paquete a Querétaro?",
    "días de entrega", "tiempos estimados", "¿Cuántas semanas tardan en llevar mis cosas?",

    # Horarios
    "¿Cuáles son sus horarios?", "¿A qué hora abren?", "¿Hasta qué hora están abiertos?",
    "¿Abren los domingos?", "¿Cuál es el horario de atención?", "horario de sucursal",
    "horarios", "a qué hora cierran", "están abiertos hoy", "¿Trabajan los sábados?",

    # Prohibidos / Artículos Especiales (Armas, etc.) - Debería caer bajo prohibiciones o dar un fallback
    "quiero enviar armas", "se pueden enviar pistolas", "¿puedo enviar dinero?", "quiero mandar joyas",
    "puedo enviar alimentos perecederos?", "¿se permite fruta o verdura?", "puedo llevar líquidos?",
    "quiero enviar medicina", "se puede mandar alcohol o cerveza?", "puedo echar cigarros al paquete?",
    
    # Dudas con artículos específicos (TV, Bicis, Camas)
    "¿Cobran extra por mandar una televisión?", "¿Puedo enviar una bicicleta?",
    "quiero mandar una lavadora y un refri", "¿cuánto por una mochila de escuela?",
    "cuánto cuesta enviar una llanta?", "se puede mandar un colchón King size?",

    # Contacto Humano
    "quiero hablar con una persona", "pueden pasarme con un asesor", "hablar con humano",
    "tengo una queja, quiero un agente", "duda que necesito hablar con alguien",
    "número para llamar", "tienen algún teléfono", "no entiendo, pásame con alguien físico",
    
    # Rastreo
    "quiero rastrear mi pedido", "dónde viene mi caja", "¿cómo sé dónde está mi paquete?",
    "tienen algún número de guía?", "cuál es el status de mi envío", "ya llegó mi caja?",
    "no han entregado mi caja, quiero saber dónde anda", "tracking", "seguir mi paquete",
    
    # Promociones / Descuentos
    "¿Tienen alguna oferta?", "¿Hay descuento si envío más de dos cajas?",
    "promos", "me haces un descuento?", "¿tienen cupones de descuento?",
    
    # Saludos y misceláneos
    "hola", "buen día", "qué tal", "buenas tardes", "hey",
    "muchas gracias", "adiós", "ok", "entendido", "hasta luego",
    "¿Cómo estás?", "gracias por la información", "luego les hablo", "bye"
]

# Completar hasta 100 variaciones si es necesario
while len(QUESTIONS) < 100:
    QUESTIONS.append(f"Pregunta variada adicional {len(QUESTIONS)}")

def run_tests():
    print(f"==============================================")
    print(f" INICIANDO TEST DEL CHATBOT: {len(QUESTIONS)} PREGUNTAS")
    print(f"==============================================")
    
    results = []
    
    for i, q in enumerate(QUESTIONS):
        res = cb.chatbot_response(q)
        # Recortar response para la tabla
        resp_trunc = res["response"].replace("\n", " ")
        if len(resp_trunc) > 50:
            resp_trunc = resp_trunc[:50] + "..."
            
        results.append([
            i+1, 
            q[:30] + ("..." if len(q) > 30 else ""), 
            res["tag"], 
            f"{res['probability']:.2f}",
            resp_trunc
        ])

    print(tabulate(results, headers=["A", "Pregunta", "Intent Detectado", "Confianza", "Respuesta del Bot"]))

if __name__ == "__main__":
    run_tests()
