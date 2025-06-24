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
load_dotenv("datos.env")

app = FastAPI()

# Modelo para entrada de predicción, agregamos opción para graficar
class EntradaPrediccion(BaseModel):
    horas_a_predecir: int
    graficar: bool = False

# Función para conectarse y obtener los datos
def obtener_datos(eliminar_id=True):
    try:
        ssh_host = os.getenv("SSH_HOST")
        ssh_port = int(os.getenv("SSH_PORT"))
        ssh_user = os.getenv("SSH_USER")
        ssh_password = os.getenv("SSH_PASSWORD")
        remote_bind_host = os.getenv("REMOTE_BIND_HOST")
        remote_bind_port = int(os.getenv("REMOTE_BIND_PORT"))
        local_bind_port = int(os.getenv("LOCAL_BIND_PORT"))
        mongo_db = os.getenv("MONGO_DB")
        mongo_collection = os.getenv("MONGO_COLLECTION")
        limite = int(os.getenv("LIMIT_DOCUMENTOS", 100))

        if not all([ssh_host, ssh_user, ssh_password, mongo_db, mongo_collection]):
            raise ValueError("Faltan variables de entorno necesarias.")

        with SSHTunnelForwarder(
            (ssh_host, ssh_port),
            ssh_username=ssh_user,
            ssh_password=ssh_password,
            remote_bind_address=(remote_bind_host, remote_bind_port),
            local_bind_address=('localhost', local_bind_port)
        ) as tunnel:

            client = MongoClient(f"mongodb://localhost:{local_bind_port}")
            db = client[mongo_db]
            collection = db[mongo_collection]

            datos = list(collection.find().limit(limite))
            df = pd.DataFrame(datos)

            if eliminar_id and "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)

            return df

    except Exception as e:
        print("Error al conectarse a MongoDB:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Error de conexión a la base de datos.")

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

        return df.to_dict(orient="records")

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

        resultados = [
            {"hora_futura": int(hora), "prediccion_valor": float(pred)}
            for hora, pred in zip(horas_futuras.flatten(), predicciones)
        ]

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
