import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Predicción de Aprobación de Curso")
st.write("Esta aplicación procesa las variables de entrada y realiza una predicción utilizando un modelo de Bagging pre-entrenado.")

# --- SECCIÓN 1: PREDICCIÓN INDIVIDUAL ---
st.header("Predicción Individual")

# Opciones para la variable Felder
opciones_felder = ['sensorial', 'activo', 'visual', 'equilibrio', 'secuencial', 'reflexivo', 'verbal', 'intuitivo']

# Entradas del formulario con Slider para la nota del examen de admisión
felder_input = st.selectbox("Selecciona el estilo de aprendizaje (Felder):", opciones_felder, key="felder_individual")
examen_input = st.slider("Examen de Admisión:", min_value=0.0, max_value=5.0, value=3.83, step=0.01, key="examen_individual")

# Crear un DataFrame temporal con los datos del usuario
data_dict = {
    'Felder': [felder_input],
    'Examen_admisión': [examen_input]
}
df_input = pd.DataFrame(data_dict)

if st.button("Realizar Predicción", key="btn_individual"):
    try:
        # Copia de trabajo
        df_procesado = df_input.copy()

        # Cargar y aplicar el transformador/columnas de One-Hot para la variable Felder
        one_hot_transformer = joblib.load('/content/one_hot_columns.joblib')

        if isinstance(one_hot_transformer, list):
            si_columnas_one_hot = [col for col in one_hot_transformer if 'Felder_' in col]
            for col_name in si_columnas_one_hot:
                valor_esperado = col_name.replace('Felder_', '')
                df_procesado[col_name] = (df_procesado['Felder'] == valor_esperado).astype(float)
        else:
            df_encoded = pd.get_dummies(df_procesado[['Felder']])
            df_procesado = pd.concat([df_procesado, df_encoded], axis=1)
            si_columnas_one_hot = [col for col in df_procesado.columns if 'Felder_' in col]

        # Eliminar la variable original Felder
        df_procesado = df_procesado.drop(columns=['Felder'], errors='ignore')

        # Asegurar que existan todas las columnas que el modelo espera
        if isinstance(one_hot_transformer, list):
            for col in one_hot_transformer:
                if col not in df_procesado.columns and col != 'Examen_admision_scaled':
                    df_procesado[col] = 0.0

        # Normalizar la variable Examen_admisión con 'min_max_scaler.joblib'
        scaler = joblib.load('/content/min_max_scaler.joblib')
        df_procesado['Examen_admision_scaled'] = scaler.transform(df_procesado[['Examen_admisión']])[0][0]
        df_procesado = df_procesado.drop(columns=['Examen_admisión'], errors='ignore')

        # Reordenar las columnas conforme lo espera el modelo
        columnas_ordenadas = one_hot_transformer
        df_procesado = df_procesado[columnas_ordenadas]

        st.subheader("Datos Procesados para el Modelo")
        st.dataframe(df_procesado)

        # Predicción con 'bagging_optimizado.joblib'
        model = joblib.load('/content/bagging_optimizado.joblib')
        prediccion = model.predict(df_procesado)

        st.success(f"La predicción del modelo (Nota Final Estimada) es: {prediccion[0]:.4f}")

    except Exception as e:
        st.error(f"Ocurrió un error durante el procesamiento o la predicción: {e}")


# --- SECCIÓN 2: CARGA DE ARCHIVOS MASIVA ---
st.write("---")
st.header("Evaluación por Archivo")
st.write("Sube un archivo Excel (.xlsx) o CSV (.csv) que contenga las columnas `Felder` y `Examen_admisión` para realizar predicciones masivas.")

archivo_subido = st.file_uploader("Selecciona un archivo", type=["xlsx", "csv"])

if archivo_subido is not None:
    try:
        # Cargar el archivo según su formato
        if archivo_subido.name.endswith('.xlsx'):
            df_archivo = pd.read_excel(archivo_subido)
        else:
            df_archivo = pd.read_csv(archivo_subido)

        # Validar columnas requeridas
        columnas_requeridas = ['Felder', 'Examen_admisión']
        if not all(col in df_archivo.columns for col in columnas_requeridas):
            st.error(f"El archivo debe contener exactamente las columnas: {columnas_requeridas}")
        else:
            st.success("¡Archivo cargado correctamente!")
            st.write("Vista previa de los datos:")
            st.dataframe(df_archivo.head())

            if st.button("Procesar y Evaluar Archivo", key="btn_masivo"):
                with st.spinner("Procesando y generando predicciones..."):
                    # Cargar los componentes del pipeline
                    one_hot_transformer = joblib.load('/content/one_hot_columns.joblib')
                    scaler = joblib.load('/content/min_max_scaler.joblib')
                    model = joblib.load('/content/bagging_optimizado.joblib')

                    resultados = []
                    
                    # Procesar fila por fila de forma segura
                    for index, fila in df_archivo.iterrows():
                        df_temp = pd.DataFrame([fila])
                        
                        # Aplicar One-Hot para Felder
                        if isinstance(one_hot_transformer, list):
                            for col in one_hot_transformer:
                                if col.startswith('Felder_'):
                                    categoria = col.replace('Felder_', '')
                                    df_temp[col] = 1.0 if str(df_temp['Felder'].iloc[0]).lower() == str(categoria).lower() else 0.0
                        
                        # Normalizar examen admisión
                        df_temp['Examen_admision_scaled'] = scaler.transform(df_temp[['Examen_admisión']])[0][0]
                        
                        # Reordenar las columnas para el modelo
                        df_temp_procesado = df_temp[one_hot_transformer]
                        
                        # Realizar predicción
                        pred = model.predict(df_temp_procesado)[0]
                        resultados.append(pred)
                    
                    # Agregar resultados al dataframe original
                    df_archivo['Nota_Final_Predicha'] = resultados
                    
                    st.subheader("Resultados Obtenidos")
                    st.dataframe(df_archivo)
                    
                    # Permitir la descarga del resultado
                    if archivo_subido.name.endswith('.xlsx'):
                        # Exportar a excel para descarga en memoria
                        import io
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            df_archivo.to_excel(writer, index=False, sheet_name='Predicciones')
                        data_descarga = output.getvalue()
                        formato_descarga = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        nombre_descarga = "predicciones_resultados.xlsx"
                    else:
                        data_descarga = df_archivo.to_csv(index=False).encode('utf-8')
                        formato_descarga = "text/csv"
                        nombre_descarga = "predicciones_resultados.csv"
                    
                    st.download_button(
                        label="Descargar resultados",
                        data=data_descarga,
                        file_name=nombre_descarga,
                        mime=formato_descarga
                    )

    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
