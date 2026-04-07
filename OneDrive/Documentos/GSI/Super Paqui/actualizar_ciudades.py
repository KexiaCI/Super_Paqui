import json
import os

def actualizar_ciudades():
    print("="*50)
    print(" ACTUALIZADOR DE CIUDADES DE SUPER PAQUI")
    print("="*50)
    print("Esta herramienta te permite reescribir el catálogo completo de ciudades.")
    print("Deberás tener un archivo llamado 'nuevas_ciudades.json' en esta misma carpeta,")
    print("con una estructura similar a esta:")
    print('''
{
  "Ciudad de México": ["Alvaro Obregón", "Coyoacán", "Cuauhtémoc"],
  "Guanajuato": ["Celaya", "Irapuato", "León", "Silao"],
  "Querétaro": ["Corregidora", "Querétaro", "San Juan del Río"]
}
    ''')
    
    file_path = "nuevas_ciudades.json"
    
    if not os.path.exists(file_path):
        print(f"⚠️ No se encontró '{file_path}'. ")
        print("Por favor, pon el archivo con tu lista de ciudades en esta carpeta y vuelve a ejecutar este script.")
        
        # Muestra cómo hacer uno de prueba
        ejemplo = {
            "Ciudad de México": ["Azcapotzalco", "Coyoacán", "Cuajimalpa", "Cuauhtémoc", "Gustavo A. Madero", "Tlalpan", "Xochimilco"],
            "Estado de México": ["Ecatepec", "Naucalpan", "Tlalnepantla", "Toluca", "Texcoco", "Cuautitlán"]
        }
        with open("ejemplo_ciudades.json", "w", encoding="utf-8") as f:
            json.dump(ejemplo, f, ensure_ascii=False, indent=2)
        print(f"👉 Te he creado un 'ejemplo_ciudades.json' para que veas el formato.")
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            nuevos_datos = json.load(f)
            
        estados = sorted(list(nuevos_datos.keys()))
        ciudades_por_estado = {}
        
        for est, ciuds in nuevos_datos.items():
            lista_limpia = sorted(list(set(ciuds))) # Eliminar duplicados y ordenar
            if "Otra Ciudad" not in lista_limpia:
                lista_limpia.append("Otra Ciudad")
            ciudades_por_estado[est] = lista_limpia
            
        final_mapping = {
            "estados": estados,
            "ciudades_por_estado": ciudades_por_estado
        }
        
        # Guardar en las dos ubicaciones
        paths_to_update = [
            "whatsapp/ciudades_mapping.json",
            "messenger/ciudades_mapping.json"
        ]
        
        for p in paths_to_update:
            if os.path.exists(os.path.dirname(p)):
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(final_mapping, f, ensure_ascii=False, indent=2)
                print(f"✅ ¡Actualizado!: {p}")
            else:
                print(f"⚠️ No se encontró la carpeta base para '{p}'.")
                
        print("\n¡Catálogo actualizado existosamente! Los cambios se reflejarán instantáneamente.")
        
    except Exception as e:
        print(f"❌ Ocurrió un error leyendo {file_path}: {e}")

if __name__ == "__main__":
    actualizar_ciudades()
