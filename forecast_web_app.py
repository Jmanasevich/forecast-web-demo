
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from io import BytesIO
import base64

st.set_page_config(page_title="Forecast & Inventario", layout="wide")

PASSWORD = "demo123"

def check_password():
    def password_entered():
        if st.session_state["password"] == PASSWORD:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Contraseña", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Contraseña", type="password", on_change=password_entered, key="password")
        st.error("Contraseña incorrecta")
        return False
    else:
        return True

if not check_password():
    st.stop()

st.title("Forecast y Planificación de Inventario por SKU")
archivo = st.file_uploader("Carga tu archivo Excel o CSV con los datos históricos", type=["csv", "xlsx"])

if archivo:
    if archivo.name.endswith(".csv"):
        df = pd.read_csv(archivo, parse_dates=['Fecha'])
    else:
        df = pd.read_excel(archivo, parse_dates=['Fecha'])

    st.success("Archivo cargado con éxito")

    columnas_necesarias = ["Fecha", "SKU", "Ventas", "Precio", "Promocion", "Dia_semana", "Es_feriado"]
    if not all(col in df.columns for col in columnas_necesarias):
        st.error(f"Tu archivo debe tener las columnas: {', '.join(columnas_necesarias)}")
        st.stop()

    df.sort_values(["SKU", "Fecha"], inplace=True)
    df["Ventas_t-1"] = df.groupby("SKU")["Ventas"].shift(1)
    df["Ventas_t-2"] = df.groupby("SKU")["Ventas"].shift(2)
    df.dropna(inplace=True)

    resultados = []
    dias_cobertura = 7
    factor_seguridad = 1.2

    for sku in df["SKU"].unique():
        data = df[df["SKU"] == sku].copy()
        X = data[["Ventas_t-1", "Ventas_t-2", "Precio", "Promocion", "Dia_semana", "Es_feriado"]]
        y = data["Ventas"]

        split = int(len(data) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        modelo = RandomForestRegressor(n_estimators=100, random_state=42)
        modelo.fit(X_train, y_train)

        pred = modelo.predict(X_test)
        forecast = data.iloc[split:].copy()
        forecast["Prediccion"] = pred
        resultados.append(forecast.tail(dias_cobertura))

    df_forecast = pd.concat(resultados).reset_index(drop=True)

    demanda = df_forecast.groupby("SKU")["Prediccion"].sum().rename("Demanda_7_dias")
    stock_objetivo = demanda * factor_seguridad
    stock_actual = pd.Series(np.random.randint(100, 200, size=len(stock_objetivo)), index=stock_objetivo.index)
    compra_sugerida = (stock_objetivo - stock_actual).clip(lower=0).round()

    df_plan = pd.DataFrame({
        "SKU": demanda.index,
        "Demanda_7_dias": demanda.values,
        "Stock_objetivo": stock_objetivo.values,
        "Stock_actual": stock_actual.values,
        "Compra_sugerida": compra_sugerida.values
    })

    st.subheader("Resumen de Planificación")
    st.dataframe(df_plan, use_container_width=True)

    def generar_descarga(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_forecast.to_excel(writer, sheet_name='Forecast', index=False)
            df_plan.to_excel(writer, sheet_name='Planificacion', index=False)
        output.seek(0)
        b64 = base64.b64encode(output.read()).decode()
        return f'<a href="data:application/octet-stream;base64,{b64}" download="forecast_resultado.xlsx">Descargar Excel</a>'

    st.markdown(generar_descarga(df_plan), unsafe_allow_html=True)
    st.success("Forecast y planificación generados exitosamente")
