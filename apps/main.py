from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from sshtunnel import SSHTunnelForwarder
from pymongo import MongoClient
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os
import traceback
import io
import base64
import matplotlib.pyplot as plt

# Cargar variables desde datos.env
load_dotenv("../datos.env")

app = FastAPI()

# Modelo para entrada de predicción, agregamos opción para graficar
class EntradaPrediccion(BaseModel):
    horas_a_predecir: int
    graficar: bool = False

# Función para conectarse y obtener los datos
def obtener_datos(eliminar_id=True):
    try:
        # Configuración para conexión local
        mongo_host = os.getenv("MONGO_HOST", "localhost")
        mongo_port = int(os.getenv("MONGO_PORT", 27017))
        mongo_db = os.getenv("MONGO_DB")
        mongo_collection = os.getenv("MONGO_COLLECTION")
        limite = int(os.getenv("LIMIT_DOCUMENTOS", 100))
        
        # Credenciales opcionales para MongoDB local
        mongo_user = os.getenv("MONGO_USER")
        mongo_password = os.getenv("MONGO_PASSWORD")
        mongo_auth_db = os.getenv("MONGO_AUTH_DB", "admin")

        # Configuración SSH comentada para uso futuro
        # ssh_host = os.getenv("SSH_HOST")
        # ssh_port = int(os.getenv("SSH_PORT"))
        # ssh_user = os.getenv("SSH_USER")
        # ssh_password = os.getenv("SSH_PASSWORD")
        # remote_bind_host = os.getenv("REMOTE_BIND_HOST")
        # remote_bind_port = int(os.getenv("REMOTE_BIND_PORT"))
        # local_bind_port = int(os.getenv("LOCAL_BIND_PORT"))

        if not all([mongo_db, mongo_collection]):
            raise ValueError("Faltan variables de entorno necesarias.")

        # Conexión local a MongoDB con autenticación opcional
        if mongo_user and mongo_password:
            # Conexión con credenciales
            client = MongoClient(f"mongodb://{mongo_user}:{mongo_password}@{mongo_host}:{mongo_port}/{mongo_auth_db}")
        else:
            # Conexión sin credenciales
            client = MongoClient(f"mongodb://{mongo_host}:{mongo_port}")

        db = client[mongo_db]
        collection = db[mongo_collection]

        datos = list(collection.find().limit(limite))
        df = pd.DataFrame(datos)

        if eliminar_id and "_id" in df.columns:
            df.drop(columns=["_id"], inplace=True)

        return df

        # Código SSH comentado para uso futuro
        # with SSHTunnelForwarder(
        #     (ssh_host, ssh_port),
        #     ssh_username=ssh_user,
        #     ssh_password=ssh_password,
        #     remote_bind_address=(remote_bind_host, remote_bind_port),
        #     local_bind_address=('localhost', local_bind_port)
        # ) as tunnel:
        #     client = MongoClient(f"mongodb://localhost:{local_bind_port}")
        #     db = client[mongo_db]
        #     collection = db[mongo_collection]
        #     datos = list(collection.find().limit(limite))
        #     df = pd.DataFrame(datos)
        #     if eliminar_id and "_id" in df.columns:
        #         df.drop(columns=["_id"], inplace=True)
        #     return df

    except Exception as e:
        print("Error al conectarse a MongoDB:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Error de conexión a la base de datos.")

# Función auxiliar para limpiar valores problemáticos en JSON
def limpiar_valores_json(df):
    """Limpia valores inf, -inf, NaN para que sean compatibles con JSON"""
    def clean_value(val):
        try:
            # Verificar si es NaN
            if pd.isna(val):
                return None
            # Verificar si es infinito
            if isinstance(val, (int, float)) and (val == float('inf') or val == float('-inf')):
                return None
            # Verificar si es un número muy grande que podría causar problemas
            if isinstance(val, (int, float)) and abs(val) > 1e308:
                return None
            return val
        except:
            return None

    # Crear una copia del DataFrame para no modificar el original
    df_clean = df.copy()
    
    # Aplicar limpieza a todas las columnas
    for col in df_clean.columns:
        df_clean[col] = df_clean[col].apply(clean_value)
    
    return df_clean

# Función para convertir DataFrame a JSON de forma segura
def dataframe_to_json_safe(df):
    """Convierte DataFrame a JSON de forma segura, manejando valores problemáticos"""
    try:
        # Primero limpiar valores problemáticos
        df_clean = limpiar_valores_json(df)
        
        # Convertir a diccionario
        records = df_clean.to_dict(orient="records")
        
        # Limpiar cada registro individualmente
        cleaned_records = []
        for record in records:
            cleaned_record = {}
            for key, value in record.items():
                try:
                    # Verificar si el valor es JSON serializable
                    if isinstance(value, (int, float)) and (pd.isna(value) or np.isinf(value)):
                        cleaned_record[key] = None
                    elif isinstance(value, (int, float)) and abs(value) > 1e308:
                        cleaned_record[key] = None
                    else:
                        cleaned_record[key] = value
                except:
                    cleaned_record[key] = None
            cleaned_records.append(cleaned_record)
        
        return cleaned_records
    except Exception as e:
        print(f"Error al convertir DataFrame a JSON: {e}")
        # Fallback: devolver lista vacía
        return []

# --- Ruta GET para obtener datos con _id como str ---
@app.get("/get_datos")
def get_datos():
    try:
        # Aquí no eliminamos _id
        df = obtener_datos(eliminar_id=False)
        if df.empty:
            raise HTTPException(status_code=404, detail="No se encontraron datos.")
        
        # Convertir _id a string para que JSON sea válido
        if "_id" in df.columns:
            df["_id"] = df["_id"].astype(str)

        # Usar función segura para convertir a JSON
        return dataframe_to_json_safe(df)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error al obtener datos: {str(e)}")

# --- Ruta POST para predecir ---
@app.post("/predecir")
def predecir(data: EntradaPrediccion):
    try:
        df = obtener_datos(eliminar_id=True)
        if df.empty:
            raise HTTPException(status_code=500, detail="No se pudieron obtener datos.")

        if "fecha" not in df.columns or "valor" not in df.columns:
            raise HTTPException(status_code=500, detail="Columnas 'fecha' o 'valor' no encontradas.")

        # Convertir columna fecha a datetime y limpiar filas inválidas
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df = df.dropna(subset=["fecha", "valor"])

        # Establecer fecha como índice para usar resample
        df = df.set_index("fecha")

        # Resamplear por hora, tomando promedio
        df = df.resample("H").mean()

        # Preparar datos para el modelo de regresión
        df = df.dropna(subset=["valor"])  # Eliminar horas sin valor
        df["hora"] = np.arange(len(df))
        X = df[["hora"]]
        y = df["valor"]

        modelo = LinearRegression()
        modelo.fit(X, y)

        horas_a_predecir = data.horas_a_predecir
        if horas_a_predecir <= 0:
            raise HTTPException(status_code=400, detail="horas_a_predecir debe ser mayor que 0.")

        horas_futuras = np.arange(len(df), len(df) + horas_a_predecir).reshape(-1, 1)
        predicciones = modelo.predict(horas_futuras)

        resultados = []
        for hora, pred in zip(horas_futuras.flatten(), predicciones):
            try:
                # Verificar si la predicción es válida
                if pd.isna(pred) or np.isinf(pred) or abs(pred) > 1e308:
                    pred_valor = None
                else:
                    pred_valor = float(pred)
                
                resultados.append({
                    "hora_futura": int(hora), 
                    "prediccion_valor": pred_valor
                })
            except:
                resultados.append({
                    "hora_futura": int(hora), 
                    "prediccion_valor": None
                })

        respuesta = {"predicciones": resultados}

        if data.graficar:
            plt.figure(figsize=(10, 5))
            plt.plot(df.index, y, label="Datos históricos")
            fechas_futuras = pd.date_range(start=df.index[-1], periods=horas_a_predecir + 1, freq="H")[1:]
            plt.plot(fechas_futuras, predicciones, label="Predicción", linestyle="--")
            plt.title("Predicción de valores")
            plt.xlabel("Fecha")
            plt.ylabel("Valor")
            plt.legend()
            plt.grid(True)

            buffer = io.BytesIO()
            plt.savefig(buffer, format="png")
            buffer.seek(0)
            imagen_base64 = base64.b64encode(buffer.read()).decode("utf-8")
            buffer.close()
            plt.close()

            respuesta["grafica"] = imagen_base64

        return respuesta

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")
