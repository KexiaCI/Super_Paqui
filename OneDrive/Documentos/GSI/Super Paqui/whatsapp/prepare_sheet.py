import os
import gspread
from google.oauth2.service_account import Credentials

def prepare_sheet():
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_path = r"c:\Users\uriel\OneDrive\Desktop\whatsapp 2\RORS890824N37.json"
        spreadsheet_id = "1zJ6NGUTtMGNvXoviYwKK4oyE70xnbF0K2Zv4iTqVdE8"
        
        creds = Credentials.from_service_account_file(creds_path, scopes=scope)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(spreadsheet_id).sheet1
        
        headers = sheet.row_values(1)
        expected_headers = [
            "Nombre", "Origen", "Destino", "Medidas", "Contenido", 
            "Teléfono", "Valor", "Fragil", "Fecha Envío", "Notas", 
            "Fecha Registro", "Cotización", "Estatus"
        ]
        
        if not headers or headers[0] != "Nombre":
            print("Writing initial headers...")
            sheet.insert_row(expected_headers, 1)
            print("Headers written.")
        elif "Estatus" not in headers:
            print("Adding 'Estatus' header...")
            sheet.update_cell(1, len(expected_headers), "Estatus")
            print("Header added successfully.")
        else:
            print("Headers are OK.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    prepare_sheet()
