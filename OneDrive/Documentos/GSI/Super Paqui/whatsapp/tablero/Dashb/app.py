import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import plotly.express as px
import os

# Configuración de la página
st.set_page_config(page_title="Dashboard Paquetería San Diego", layout="wide")

# Estilos personalizados (Púrpura y Naranja)
st.markdown("""
<style>
    .main {
        background-color: #f4f4f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: #000000 !important;
    }
    .stMetric label, .stMetric [data-testid="stMetricValue"] {
        color: #000000 !important;
    }
    h1, h2, h3 {
        color: #3f2a8d;
    }
</style>
""", unsafe_allow_html=True)

# Conexión a Google Sheets
def load_data():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds_path = os.path.join(os.path.dirname(__file__), "RORS890824N37.json")
    creds = Credentials.from_service_account_file(creds_path, scopes=scope)
    client = gspread.authorize(creds)
    spreadsheet_id = "1zJ6NGUTtMGNvXoviYwKK4oyE70xnbF0K2Zv4iTqVdE8"
    sheet = client.open_by_key(spreadsheet_id).sheet1
    
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    
    # Limpieza de datos
    if not df.empty:
        # Limpiar columna de cotización (quitar '$', ' USD' y convertir a numérico)
        if 'Cotización' in df.columns:
            df['Precio Num'] = df['Cotización'].str.replace('$', '').str.replace(' USD', '').str.replace(',', '').apply(pd.to_numeric, errors='coerce')
        
        # Convertir fecha a datetime
        if 'Fecha de registro' in df.columns:
            df['Fecha de registro'] = pd.to_datetime(df['Fecha de registro'], errors='coerce')
            
    return df

st.title("🚚 Dashboard de Negocios - Paquetería San Diego")
st.markdown("### Análisis interactivo de pedidos y tendencias")

df = load_data()

if df.empty:
    st.warning("No hay datos registrados todavía. El dashboard se actualizará cuando caigan los primeros pedidos.")
else:
    # --- FILTROS LATERALES ---
    st.sidebar.header("Filtros")
    origins = st.sidebar.multiselect("Origen", options=df['Origen'].unique(), default=df['Origen'].unique())
    destinations = st.sidebar.multiselect("Destino", options=df['Destino'].unique(), default=df['Destino'].unique())
    
    df_filtered = df[(df['Origen'].isin(origins)) & (df['Destino'].isin(destinations))]

    # --- MÉTRICAS CLAVE ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Pedidos", len(df_filtered))
    with col2:
        total_ingresos = df_filtered['Precio Num'].sum()
        st.metric("Ingresos Estimados", f"${total_ingresos:,.2f} USD")
    with col3:
        avg_ticket = df_filtered['Precio Num'].mean()
        st.metric("Ticket Promedio", f"${avg_ticket:,.2f} USD")
    with col4:
        fragile_pct = (df_filtered['Fragil'] == 'Sí').mean() * 100
        st.metric("% Frágil", f"{fragile_pct:.1f}%")

    st.divider()

    # --- GRÁFICOS ---
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.subheader("📍 Volumen por Destino")
        fig_dest = px.bar(df_filtered['Destino'].value_counts().reset_index(), 
                         x='Destino', y='count', 
                         color_discrete_sequence=['#f27121'],
                         labels={'count': 'Número de Pedidos', 'index': 'Ciudad'})
        st.plotly_chart(fig_dest, use_container_width=True)

    with row1_col2:
        st.subheader("📦 Tipos de Medidas más Populares")
        fig_meas = px.pie(df_filtered, names='Medida', 
                         color_discrete_sequence=px.colors.qualitative.Prism)
        st.plotly_chart(fig_meas, use_container_width=True)

    st.subheader("📅 Tendencia de Pedidos por Día")
    if 'Fecha de registro' in df_filtered.columns:
        df_trend = df_filtered.groupby(df_filtered['Fecha de registro'].dt.date).size().reset_index(name='pedidos')
        fig_trend = px.line(df_trend, x='Fecha de registro', y='pedidos', 
                           markers=True, color_discrete_sequence=['#3f2a8d'])
        st.plotly_chart(fig_trend, use_container_width=True)

    st.divider()

    # --- CONCLUSIONES (AI INSIGHTS) ---
    st.subheader("🔍 Conclusiones y Recomendaciones")
    
    # Lógica de conclusiones automatizada
    top_dest = df_filtered['Destino'].mode()[0] if not df_filtered['Destino'].empty else "N/A"
    top_origin = df_filtered['Origen'].mode()[0] if not df_filtered['Origen'].empty else "N/A"
    top_measure = df_filtered['Medida'].mode()[0] if not df_filtered['Medida'].empty else "N/A"
    
    col_ins1, col_ins2 = st.columns(2)
    
    with col_ins1:
        st.info(f"**Destino Estrella:** Su ruta más fuerte actualmente es hacia **{top_dest}**. Podría considerar una promoción especial o publicidad enfocada en esta zona.")
        st.info(f"**Origen Principal:** La mayoría de sus clientes están enviando desde **{top_origin}**.")
        
    with col_ins2:
        st.success(f"**Producto Líder:** La medida **{top_measure}** es la más solicitada. Asegúrese de tener suficiente stock de este tipo de cajas.")
        if fragile_pct > 30:
            st.warning("⚠️ **Alerta de Seguridad:** Más del 30% de sus envíos son frágiles. Considere revisar sus materiales de empaque.")

    # --- TABLA DE DATOS RECIENTES ---
    with st.expander("Ver lista de pedidos recientes"):
        st.dataframe(df_filtered.sort_values(by='Fecha de registro', ascending=False), use_container_width=True)

# Botón de actualización
if st.button('🔄 Actualizar Datos'):
    st.cache_data.clear()
    st.rerun()
