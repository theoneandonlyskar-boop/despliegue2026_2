import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Configurar la página de Streamlit
st.set_page_config(page_title="Predicción de Aprobación de Curso", layout="wide")

st.title("Predicción de Nota Final - Curso")
st.write("Estime la nota final de los estudiantes de forma individual o cargando un archivo con múltiples registros utilizando el modelo de Bagging.")

# Obtener la ruta base del directorio donde se ejecuta el script para garantizar portabilidad en GitHub
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()

# 1. Cargar artefactos necesarios de forma segura
@st.cache_resource
def load_artifacts():
    try:
        columnas_one_hot = joblib.load(os.path.join(BASE_DIR, 'one_hot_columns.joblib'))
        scaler = joblib.load(os.path.join(BASE_DIR, 'min_max_scaler.joblib'))
        model = joblib.load(os.path.join(BASE_DIR, 'bagging_optimizado.joblib'))
        return columnas_one_hot, scaler, model
    except Exception as e:
        st.error(f"Error al cargar los archivos .joblib: {e}")
        return None, None, None

columnas_one_hot, scaler, model = load_artifacts()

if columnas_one_hot and scaler and model:
    # Crear pestañas para organizar las predicciones
    tab1, tab2 = st.tabs(["Predicción Individual", "Predicción por Archivo (Masiva)"])

    # Extraer las categorías posibles para la variable 'Felder'
    categorias_felder = [col.replace('Felder_', '') for col in columnas_one_hot if col.startswith('Felder_')]

    with tab1:
        st.header("Datos del Estudiante")
        felder_selected = st.selectbox("Estilo de Aprendizaje (Felder)", options=categorias_felder)
        examen_admision = st.slider("Nota de Examen de Admisión", min_value=0.0, max_value=5.0, value=3.8, step=0.05)

        if st.button("Calcular Predicción Individual"):
            df_input = pd.DataFrame([{'Felder': felder_selected, 'Examen_admisión': examen_admision}])

            # Codificación One-Hot
            for col in columnas_one_hot:
                if col.startswith('Felder_'):
                    categoria = col.replace('Felder_', '')
                    df_input[col] = 1.0 if felder_selected == categoria else 0.0

            # Escalado de Examen de Admisión
            df_input['Examen_admision_scaled'] = scaler.transform(df_input[['Examen_admisión']])[0][0]

            # Selección y orden de columnas
            columnas_finales = [col for col in columnas_one_hot if col in df_input.columns]
            df_procesado = df_input[columnas_finales]

            prediccion = model.predict(df_procesado)[0]
            st.success(f"### Nota Final Estimada: {prediccion:.3f}")

    with tab2:
        st.header("Predicción Masiva a través de Archivo")
        st.write("Sube un archivo en formato Excel (.xlsx) o CSV (.csv). El archivo debe contener al menos las columnas **'Felder'** y **'Examen_admisión'**.")
        
        uploaded_file = st.file_uploader("Seleccionar archivo", type=['csv', 'xlsx'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_original = pd.read_csv(uploaded_file)
                else:
                    df_original = pd.read_excel(uploaded_file)
                
                st.write("### Vista previa de los datos subidos:")
                st.dataframe(df_original.head())
                
                # Validar columnas necesarias
                if 'Felder' in df_original.columns and 'Examen_admisión' in df_original.columns:
                    if st.button("Procesar y Predecir"):
                        df_batch = df_original.copy()
                        
                        # Aplicar codificación One-Hot masiva para 'Felder'
                        for col in columnas_one_hot:
                            if col.startswith('Felder_'):
                                categoria = col.replace('Felder_', '')
                                df_batch[col] = df_batch['Felder'].apply(lambda x: 1.0 if str(x).strip() == categoria else 0.0)
                        
                        # Aplicar el Escalador de forma masiva
                        valores_admision = df_batch[['Examen_admisión']].values
                        df_batch['Examen_admision_scaled'] = scaler.transform(valores_admision)[:, 0]
                        
                        # Alinear las columnas con el modelo
                        columnas_finales = [col for col in columnas_one_hot if col in df_batch.columns]
                        df_procesado_batch = df_batch[columnas_finales]
                        
                        # Generar predicciones
                        predicciones_batch = model.predict(df_procesado_batch)
                        
                        # Añadir la predicción al dataframe original expuesto al usuario
                        df_resultado = df_original.copy()
                        df_resultado['Nota_final_Predicha'] = predicciones_batch
                        
                        st.success("¡Predicciones generadas con éxito!")
                        st.write("### Resultados:")
                        st.dataframe(df_resultado)
                        
                        # Permitir descargar los resultados
                        csv_data = df_resultado.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Descargar resultados en CSV",
                            data=csv_data,
                            file_name="predicciones_estudiantes.csv",
                            mime="text/csv"
                        )
                else:
                    st.error("El archivo cargado debe incluir obligatoriamente las columnas 'Felder' y 'Examen_admisión'.")
            except Exception as e:
                st.error(f"Error al procesar el archivo: {e}")
else:
    st.warning("Por favor, asegúrate de que los archivos 'one_hot_columns.joblib', 'min_max_scaler.joblib' y 'bagging_optimizado.joblib' se encuentren en la misma carpeta que este script.")
