import streamlit as st
import joblib
import pickle
import numpy as np
import pandas as pd
import psycopg2
from datetime import datetime

# Fetch variables
USER = "postgres.qzwspxajgjywkwudmudc" #os.getenv("user")
PASSWORD = "MF$a.081204."# os.getenv("password")
HOST = "aws-1-us-west-2.pooler.supabase.com" #os.getenv("host")
PORT = "5432" #os.getenv("port")
DBNAME = "postgres" #os.getenv("dbname")

# Configuración de la página
st.set_page_config(page_title="Predictor de Iris", page_icon="🌸")

# Función para conectar a la base de datos
def get_db_connection():
    try:
        connection = psycopg2.connect(
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT,
            dbname=DBNAME
        )
        return connection
    except Exception as e:
        st.error(f"Error conectando a la base de datos: {str(e)}")
        return None

# Función para insertar predicción en tb_iris
def insert_prediction(l_s, a_s, l_p, a_p, prediccion):
    """Inserta los valores y la predicción en la tabla tb_iris
    l_s = largo del sépalo
    a_s = ancho del sépalo
    l_p = largo del pétalo
    a_p = ancho del pétalo
    """
    conn = get_db_connection()
    if conn is None:
        return False
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO public.tb_iris (l_s, a_s, l_p, a_p, prediccion)
            VALUES (%s, %s, %s, %s, %s)
        """, (l_s, a_s, l_p, a_p, prediccion))
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error insertando predicción: {str(e)}")
        return False

# Función para obtener el histórico ordenado DESC
def get_predictions_history():
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, created_at, l_s, a_s, l_p, a_p, prediccion
            FROM public.tb_iris
            ORDER BY created_at DESC
            LIMIT 50
        """)
        data = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if data:
            df = pd.DataFrame(data, columns=['ID', 'Fecha', 'Largo Sépalo (l_s)', 
                                             'Ancho Sépalo (a_s)', 'Largo Pétalo (l_p)', 
                                             'Ancho Pétalo (a_p)', 'Predicción'])
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error obteniendo histórico: {str(e)}")
        return pd.DataFrame()



# Función para cargar los modelos
@st.cache_resource
def load_models():
    try:
        model = joblib.load('components/iris_model.pkl')
        scaler = joblib.load('components/iris_scaler.pkl')
        with open('components/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        return model, scaler, model_info
    except FileNotFoundError:
        st.error("No se encontraron los archivos del modelo en la carpeta 'models/'")
        return None, None, None

# Título
st.title("🌸 Predictor de Especies de Iris")

# Cargar modelos
model, scaler, model_info = load_models()

if model is not None:
    # Inputs
    st.header("Ingresa las características de la flor:")
    
    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.number_input("Largo del Sépalo (l_s) en cm", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
        petal_length = st.number_input("Largo del Pétalo (l_p) en cm", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
    
    with col2:
        sepal_width = st.number_input("Ancho del Sépalo (a_s) en cm", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
        petal_width = st.number_input("Ancho del Pétalo (a_p) en cm", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    
    # Botón de predicción
    if st.button("🔮 Predecir Especie"):
        # Preparar datos
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Estandarizar
        features_scaled = scaler.transform(features)
        
        # Predecir
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Mostrar resultado
        target_names = model_info['target_names']
        predicted_species = target_names[prediction]
        
        st.success(f"Especie predicha: **{predicted_species}**")
        st.write(f"Confianza: **{max(probabilities):.1%}**")
        
        # Mostrar todas las probabilidades
        with st.expander("Ver todas las probabilidades"):
            for species, prob in zip(target_names, probabilities):
                st.write(f"- {species}: {prob:.1%}")
        
        # 📝 INSERTAR EN LA BASE DE DATOS
        if insert_prediction(sepal_length, sepal_width, petal_length, petal_width, predicted_species):
            st.info("✅ Predicción guardada en la base de datos")
        else:
            st.error("❌ Error al guardar la predicción")
    
    # 📊 MOSTRAR HISTÓRICO ORDENADO DESC
    st.divider()
    st.header("📊 Histórico de Predicciones")
    
    df_history = get_predictions_history()
    if not df_history.empty:
        st.dataframe(df_history, use_container_width=True)
        
        # Estadísticas
        st.subheader("📈 Estadísticas")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de predicciones", len(df_history))
        with col2:
            from collections import Counter
            species_counts = Counter(df_history['Predicción'])
            most_common = species_counts.most_common(1)[0][0]
            st.metric("Especie más frecuente", most_common)
        with col3:
            st.metric("Primera predicción", str(df_history['Fecha'].iloc[-1]).split(" ")[0])
    else:
        st.info("No hay predicciones guardadas aún")
    
