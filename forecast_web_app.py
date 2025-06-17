import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from io import BytesIO
import base64
from fpdf import FPDF
import matplotlib.pyplot as plt
import tempfile

st.set_page_config(page_title="Forecast & Inventario", layout="wide")

PASSWORD = "alb2025"

LOGO_PATH = "logo 2.jpg"  # Logo personalizado de ALB Consultores

class PDFConLogo(FPDF):
    def header(self):
        self.image(LOGO_PATH, 10, 8, 33)  # logo empresa
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Reporte Inteligente de Inventario', border=False, ln=1, align='C')
        self.ln(10)

st.image(LOGO_PATH, width=100)

# Entrenamiento seguro filtrando NaNs en ventas
def entrenar_modelo(df):
    df_entrenamiento = df[df["Ventas"].notna()]
    X = df_entrenamiento[["Precio", "Promocion", "Dia_semana", "Es_feriado"]]
    y = df_entrenamiento["Ventas"]

    modelo = RandomForestRegressor()
    modelo.fit(X, y)
    return modelo

# Validación previa a concatenar resultados
def concatenar_resultados(resultados):
    if not resultados:
        st.error("No se pudo generar forecast para ningún SKU. Revisa que tus datos tengan suficientes datos históricos de ventas.")
        st.stop()
    return pd.concat(resultados).reset_index(drop=True)

def generar_excel():
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        if 'df_forecast' not in globals() or df_forecast.empty:
            st.error("No se pudo generar forecast para ningún SKU. Revisa que tus datos tengan valores en la columna 'Ventas'.")
            st.stop()

        df_forecast.to_excel(writer, sheet_name='Forecast', index=False)
        df_plan.to_excel(writer, sheet_name='Planificacion', index=False)

        workbook = writer.book
        resumen_hoja = workbook.add_worksheet("Resumen_Graficos")
        resumen_hoja.write(0, 0, "Gráficos de Forecast por SKU")

        row_offset = 2
        col_offset = 0

        for idx, sku in enumerate(df_forecast["SKU"].unique()):
            df_plot = df_forecast[df_forecast["SKU"] == sku].copy()

            # Corte basado en última fecha con venta real
            fecha_corte = df_plot.loc[df_plot['Ventas'].notna(), 'Fecha'].max()
            df_plot['Ventas'] = df_plot.apply(lambda row: row['Ventas'] if row['Fecha'] <= fecha_corte else None, axis=1)
            df_plot['Prediccion'] = df_plot.apply(lambda row: row['Prediccion'] if row['Fecha'] > fecha_corte else None, axis=1)

            hoja = workbook.add_worksheet(f"Graf_{sku[:25]}")
            chart = workbook.add_chart({'type': 'line'})

            hoja.write_column(0, 0, ['Fecha'] + list(df_plot['Fecha'].dt.strftime('%Y-%m-%d')))
            hoja.write_column(0, 1, ['Ventas'] + [None if pd.isna(v) else v for v in df_plot['Ventas']])
            hoja.write_column(0, 2, ['Forecast'] + [None if pd.isna(f) else f for f in df_plot['Prediccion']])

            n = len(df_plot)
            chart.add_series({
                'name': 'Ventas reales',
                'categories': [f'Graf_{sku[:25]}', 1, 0, n, 0],
                'values':     [f'Graf_{sku[:25]}', 1, 1, n, 1],
                'gap': True,
                'connect_gaps': False
            })
            chart.add_series({
                'name': 'Forecast ajustado',
                'categories': [f'Graf_{sku[:25]}', 1, 0, n, 0],
                'values':     [f'Graf_{sku[:25]}', 1, 2, n, 2],
                'gap': True,
                'connect_gaps': False
            })
            chart.set_title({'name': f"{sku}"})
            chart.set_x_axis({'name': 'Fecha'})
            chart.set_y_axis({'name': 'Unidades'})
            hoja.insert_chart('E2', chart)

            fechas_forecast = df_plot.loc[df_plot['Prediccion'].notna(), 'Fecha']
            if not fechas_forecast.empty:
                fecha_ini = fechas_forecast.min().strftime('%Y-%m-%d')
                fecha_fin = fechas_forecast.max().strftime('%Y-%m-%d')
            else:
                fecha_ini = fecha_fin = "Sin datos de forecast"

            comentario = f"SKU: {sku} | Forecast desde {fecha_ini} hasta {fecha_fin}"

            resumen_hoja.insert_chart(row_offset, col_offset, chart, {'x_scale': 0.6, 'y_scale': 0.6})
            resumen_hoja.write(row_offset + 14, col_offset, comentario)
            resumen_hoja.write(row_offset + 15, col_offset, f"Revisar tendencia y compras sugeridas")
            col_offset += 8
            if col_offset > 15:
                col_offset = 0
                row_offset += 18

    output.seek(0)
    b64 = base64.b64encode(output.read()).decode()
    enlace = f'<a href="data:application/octet-stream;base64,{b64}" download="reporte_inteligente_forecast.xlsx">Descargar reporte de forecast e inventario en Excel</a>'
    st.markdown(enlace, unsafe_allow_html=True)
    return enlace
