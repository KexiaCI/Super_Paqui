import pandas as pd
import json

df = pd.read_excel('Destino_especiales.xlsx')
mapping = df.groupby('Estado')['Destino'].apply(list).to_dict()

# Parse the states
cleaned_mapping = {}
for k, v in mapping.items():
    if k == 'EDO MEX':
        new_k = 'Estado de México'
    elif k == 'CDMX' or k == 'CIUDAD DE MEXICO' or k == 'DIF' or k == 'CIUDAD DE MÉXICO':
        new_k = 'Ciudad de México'
    else:
        new_k = k.title()
    cleaned_mapping[new_k] = sorted(list(set(v)))

if 'Ciudad de México' not in cleaned_mapping:
    cleaned_mapping['Ciudad de México'] = []
    
# Add special cities to their respective states
if 'Guanajuato' not in cleaned_mapping:
    cleaned_mapping['Guanajuato'] = []
cleaned_mapping['Guanajuato'].extend(['Dolores Hidalgo', 'San Diego de la Unión', 'San Luis de la Paz', 'Guanajuato'])

if 'Querétaro' not in cleaned_mapping:
    cleaned_mapping['Querétaro'] = []
cleaned_mapping['Querétaro'].append('Querétaro')

if 'San Luis Potosí' not in cleaned_mapping:
    cleaned_mapping['San Luis Potosí'] = []
cleaned_mapping['San Luis Potosí'].append('San Luis Potosí')

# Remove duplicates
for k in cleaned_mapping:
    # Add 'Otra Ciudad' at the end
    cities = sorted(list(set(cleaned_mapping[k])))
    if 'Otra Ciudad' in cities:
        cities.remove('Otra Ciudad')
    cities.append('Otra Ciudad')
    cleaned_mapping[k] = cities

all_32_states = [
    "Aguascalientes", "Baja California", "Baja California Sur", "Campeche", "Chiapas", "Chihuahua",
    "Ciudad de México", "Coahuila", "Colima", "Durango", "Estado de México", "Guanajuato", "Guerrero",
    "Hidalgo", "Jalisco", "Michoacán", "Morelos", "Nayarit", "Nuevo León", "Oaxaca", "Puebla",
    "Querétaro", "Quintana Roo", "San Luis Potosí", "Sinaloa", "Sonora", "Tabasco", "Tamaulipas",
    "Tlaxcala", "Veracruz", "Yucatán", "Zacatecas"
]

for state in all_32_states:
    if state not in cleaned_mapping:
        cleaned_mapping[state] = ['Otra Ciudad']

output = {
    "estados": all_32_states,
    "ciudades_por_estado": cleaned_mapping
}

with open('ciudades_mapping.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("Mapping generated in ciudades_mapping.json")
