import os
import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

app = FastAPI(title="Tablero de Control - Paquetería San Diego")

# Configuración de Google Sheets
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
# La credencial está en el directorio padre
CREDS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "RORS890824N37.json")
SPREADSHEET_ID = "1zJ6NGUTtMGNvXoviYwKK4oyE70xnbF0K2Zv4iTqVdE8"

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

def get_sheet():
    # Priority 1: Read from Environment Variable (for Hugging Face/Deployment)
    service_account_info = os.environ.get("GSPREAD_SERVICE_ACCOUNT")
    
    if service_account_info:
        try:
            # Parse the JSON string from the environment variable
            info = json.loads(service_account_info)
            creds = Credentials.from_service_account_info(info, scopes=SCOPE)
            client = gspread.authorize(creds)
            return client.open_by_key(SPREADSHEET_ID).sheet1
        except Exception as e:
            print(f"Error loading credentials from Environment Variable: {e}")
            # Fallback to file if env var parsing fails
            
    # Priority 2: Read from local File (for Local Development)
    if not os.path.exists(CREDS_PATH):
        raise Exception(f"Credenciales no encontradas ni en variable de entorno ni en {CREDS_PATH}")
    creds = Credentials.from_service_account_file(CREDS_PATH, scopes=SCOPE)
    client = gspread.authorize(creds)
    return client.open_by_key(SPREADSHEET_ID).sheet1

@app.get("/", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/orders")
async def get_orders():
    try:
        sheet = get_sheet()
        all_values = sheet.get_all_values()
        if not all_values:
            return JSONResponse({"orders": []})
        
        # Consistent headers mapping as written by the chatbot
        expected_headers = [
            "Nombre", "Origen", "Destino", "Medidas", "Contenido", 
            "Teléfono", "Valor", "Fragil", "Fecha Envío", "Notas", 
            "Fecha Registro", "Cotización", "Estatus"
        ]
        
        # Detect if first row is headers or data
        has_headers = all_values[0][0] == "Nombre"
        rows = all_values[1:] if has_headers else all_values
        
        orders = []
        for idx, row in enumerate(rows):
            order = {}
            for i, header in enumerate(expected_headers):
                val = row[i] if i < len(row) else ""
                order[header] = val
            
            # Use original row index (if has_headers, data starts at row 2)
            order["row_index"] = idx + (2 if has_headers else 1)
            
            if not order.get("Estatus"):
                order["Estatus"] = "Solicitado"
                
            orders.append(order)
            
        # Ordenar por Fecha Registro descendente (Col index 10)
        try:
            orders.sort(key=lambda x: x.get("Fecha Registro", ""), reverse=True)
        except:
            pass
            
        return JSONResponse({"orders": orders})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/orders/update")
async def update_order_status(request: Request):
    try:
        data = await request.json()
        row_index = data.get("row_index")
        new_status = data.get("status")
        
        if not row_index or not new_status:
            raise HTTPException(status_code=400, detail="row_index and status are required")
            
        sheet = get_sheet()
        headers = sheet.row_values(1)
        try:
            status_col = headers.index("Estatus") + 1
        except ValueError:
            # If not found, it's column 13 based on save_to_sheets structure
            status_col = 13
            sheet.update_cell(1, status_col, "Estatus")
            
        sheet.update_cell(row_index, status_col, new_status)
        return JSONResponse({"success": True})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
