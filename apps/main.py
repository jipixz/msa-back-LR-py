from fastapi import FastAPI, HTTPException, Body, Query
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
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables desde datos.env
load_dotenv("../datos.env")

app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambia esto a tu dominio en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

        # Contar total de documentos
        total_docs = collection.count_documents({})
        logger.info(f"Total de documentos en la colección: {total_docs}")

        # Obtener documentos ordenados por fecha (más recientes primero)
        datos = list(collection.find().sort("fecha", -1).limit(limite))
        logger.info(f"Documentos obtenidos: {len(datos)}")
        
        if datos:
            # Mostrar información sobre el rango de fechas
            fechas = [doc.get('fecha') for doc in datos if 'fecha' in doc]
            if fechas:
                fechas.sort()
                logger.info(f"Rango de fechas: {fechas[0]} a {fechas[-1]}")
            
            # Mostrar campos disponibles en los primeros documentos
            campos_disponibles = set()
            for doc in datos[:5]:  # Primeros 5 documentos
                campos_disponibles.update(doc.keys())
            logger.info(f"Campos disponibles en los datos: {list(campos_disponibles)}")

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

# --- NUEVA FUNCIÓN GENERALIZADA DE PREDICCIÓN POR CAMPO ---
def predecir_campo(nombre_campo, horas_a_predecir, graficar=False):
    try:
        logger.info(f"Iniciando predicción para campo: {nombre_campo}")
        df = obtener_datos(eliminar_id=True)
        
        if df.empty:
            logger.error("DataFrame vacío obtenido de la base de datos")
            raise HTTPException(status_code=500, detail="No se pudieron obtener datos.")

        logger.info(f"Columnas originales en DataFrame: {list(df.columns)}")
        logger.info(f"Primeras filas del DataFrame: {df.head().to_dict()}")

        # Mapear campo del frontend al campo real de la BD
        campo_real = nombre_campo
        
        # Filtrar solo registros con la estructura correcta (ignorar registros con campo 'valor')
        campos_esperados = ['humedadSuelo', 'temperaturaBME', 'humedadAire', 'lluvia', 'presion', 'luminosidad']
        df_filtrado = df[df.columns.intersection(campos_esperados + ['fecha', '__v'])]
        
        # Eliminar registros que solo tengan 'valor' (datos antiguos de prueba)
        if 'valor' in df.columns:
            df_filtrado = df_filtrado.drop(columns=['valor'], errors='ignore')
            logger.info("Eliminando registros con campo 'valor' (datos antiguos de prueba)")
        
        if campo_real not in df_filtrado.columns:
            logger.error(f"Columna '{campo_real}' no encontrada en datos recientes. Columnas disponibles: {list(df_filtrado.columns)}")
            raise HTTPException(status_code=500, detail=f"Columna '{campo_real}' no encontrada en los datos recientes.")
        
        # Usar el DataFrame filtrado
        df = df_filtrado
        logger.info(f"Usando DataFrame filtrado con columnas: {list(df.columns)}")

        if "fecha" not in df.columns:
            logger.error(f"Columna 'fecha' no encontrada. Columnas disponibles: {list(df.columns)}")
            raise HTTPException(status_code=500, detail="Columna 'fecha' no encontrada en los datos.")

        # Convertir columna fecha a datetime y limpiar filas inválidas
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df = df.dropna(subset=["fecha"])
        
        if df.empty:
            logger.error("No quedaron datos válidos después de limpiar fechas")
            raise HTTPException(status_code=500, detail="No hay datos válidos para predicción.")

        # Ordenar por fecha (más recientes primero)
        df = df.sort_values('fecha', ascending=False)

        # Imputar valores faltantes de la columna a predecir usando la mediana de los valores existentes
        if df[campo_real].isnull().any():
            mediana = df[campo_real].median()
            logger.info(f"Imputando valores faltantes en '{campo_real}' usando la mediana: {mediana}")
            df[campo_real] = df[campo_real].fillna(mediana)

        # Eliminar filas que aún tengan NaN en la columna a predecir (por si la mediana no se pudo calcular)
        df = df.dropna(subset=[campo_real])
        if df.empty:
            logger.error("No hay datos válidos para predicción después de imputar valores")
            raise HTTPException(status_code=500, detail="No hay datos válidos para predicción.")

        logger.info(f"Datos válidos después de limpieza e imputación: {len(df)} filas")

        # Establecer fecha como índice para usar resample
        df = df.set_index("fecha")
        df = df.resample("H").mean()
        df = df.dropna(subset=[campo_real])
        if df.empty:
            logger.error("No quedaron datos después del resample por hora")
            raise HTTPException(status_code=500, detail="No hay suficientes datos para predicción.")
        logger.info(f"Datos después de resample: {len(df)} filas")

        # Ordenar por fecha (más antiguos primero para el modelo)
        df = df.sort_index()

        df["hora"] = np.arange(len(df))
        X = df[["hora"]]
        y = df[campo_real]

        modelo = LinearRegression()
        modelo.fit(X, y)

        if horas_a_predecir <= 0:
            raise HTTPException(status_code=400, detail="horas_a_predecir debe ser mayor que 0.")

        horas_futuras = np.arange(len(df), len(df) + horas_a_predecir).reshape(-1, 1)
        predicciones = modelo.predict(horas_futuras)

        resultados = []
        for hora, pred in zip(horas_futuras.flatten(), predicciones):
            try:
                if pd.isna(pred) or np.isinf(pred) or abs(pred) > 1e308:
                    pred_valor = None
                else:
                    pred_valor = float(pred)
                    
                    # Normalizar valores según el tipo de campo
                    if nombre_campo == 'temperaturaBME':
                        # Temperatura entre 20-40°C
                        pred_valor = max(20, min(40, pred_valor))
                    elif nombre_campo == 'humedadAire':
                        # Humedad entre 30-90%
                        pred_valor = max(30, min(90, pred_valor))
                    elif nombre_campo == 'lluvia':
                        # Lluvia entre 0-50mm
                        pred_valor = max(0, min(50, pred_valor))
                    elif nombre_campo == 'humedadSuelo':
                        # Humedad suelo entre 20-80%
                        pred_valor = max(20, min(80, pred_valor))
                
                # Calcular fecha para el día
                fecha_prediccion = df.index[-1] + pd.Timedelta(hours=int(hora) - len(df))
                dia = fecha_prediccion.strftime('%Y-%m-%d')
                
                resultados.append({
                    "hora_futura": int(hora),
                    "prediccion_valor": pred_valor,
                    "predicted": pred_valor,  # Campo que espera el frontend
                    "day": dia,  # Campo que espera el frontend
                    "actual": None  # Campo que espera el frontend
                })
            except Exception as e:
                logger.warning(f"Error procesando predicción: {e}")
                resultados.append({
                    "hora_futura": int(hora),
                    "prediccion_valor": None,
                    "predicted": None,
                    "day": None,
                    "actual": None
                })

        respuesta = {"predictions": resultados}
        logger.info(f"Predicción completada exitosamente para {nombre_campo}")

        if graficar:
            try:
                plt.figure(figsize=(10, 5))
                plt.plot(df.index, y, label="Datos históricos")
                fechas_futuras = pd.date_range(start=df.index[-1], periods=horas_a_predecir + 1, freq="H")[1:]
                plt.plot(fechas_futuras, predicciones, label="Predicción", linestyle="--")
                plt.title(f"Predicción de {nombre_campo}")
                plt.xlabel("Fecha")
                plt.ylabel(nombre_campo)
                plt.legend()
                plt.grid(True)
                buffer = io.BytesIO()
                plt.savefig(buffer, format="png")
                buffer.seek(0)
                imagen_base64 = base64.b64encode(buffer.read()).decode("utf-8")
                buffer.close()
                plt.close()
                respuesta["grafica"] = imagen_base64
            except Exception as e:
                logger.error(f"Error generando gráfica: {e}")

        return respuesta

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error inesperado en predecir_campo: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error interno en predicción: {str(e)}")

# --- RUTAS DE PREDICCIÓN PARA EL FRONTEND ---
from fastapi import Body

@app.api_route("/predictions/temperature", methods=["GET", "POST", "OPTIONS"])
def predict_temperature(
    days: int = Query(7, description="Número de días a predecir"),
    payload: dict = Body(None)
):
    try:
        if payload:
            # POST request
            horas = payload.get("hours_to_predict", 168)
            graficar = payload.get("include_graph", False)
        else:
            # GET request
            horas = days * 24
            graficar = False
        return predecir_campo("temperaturaBME", horas, graficar)
    except Exception as e:
        logger.error(f"Error en predict_temperature: {e}")
        raise

@app.api_route("/predictions/humidity", methods=["GET", "POST", "OPTIONS"])
def predict_humidity(
    days: int = Query(7, description="Número de días a predecir"),
    payload: dict = Body(None)
):
    try:
        if payload:
            # POST request
            horas = payload.get("hours_to_predict", 168)
            graficar = payload.get("include_graph", False)
        else:
            # GET request
            horas = days * 24
            graficar = False
        return predecir_campo("humedadAire", horas, graficar)
    except Exception as e:
        logger.error(f"Error en predict_humidity: {e}")
        raise

@app.api_route("/predictions/rainfall", methods=["GET", "POST", "OPTIONS"])
def predict_rainfall(
    days: int = Query(7, description="Número de días a predecir"),
    payload: dict = Body(None)
):
    try:
        if payload:
            # POST request
            horas = payload.get("hours_to_predict", 168)
            graficar = payload.get("include_graph", False)
        else:
            # GET request
            horas = days * 24
            graficar = False
        return predecir_campo("lluvia", horas, graficar)
    except Exception as e:
        logger.error(f"Error en predict_rainfall: {e}")
        raise

@app.api_route("/predictions/soil-moisture", methods=["GET", "POST", "OPTIONS"])
def predict_soil_moisture(
    days: int = Query(7, description="Número de días a predecir"),
    payload: dict = Body(None)
):
    try:
        if payload:
            # POST request
            horas = payload.get("hours_to_predict", 168)
            graficar = payload.get("include_graph", False)
        else:
            # GET request
            horas = days * 24
            graficar = False
        return predecir_campo("humedadSuelo", horas, graficar)
    except Exception as e:
        logger.error(f"Error en predict_soil_moisture: {e}")
        raise

# Ruta de ejemplo para /predictions/alerts
@app.get("/predictions/alerts")
def get_alerts():
    # Ejemplo de alerta, puedes personalizar la lógica
    return {
        "alerts": [
            {
                "type": "temperature",
                "severity": "warning",
                "message": "Temperatura alta prevista para los próximos días.",
                "recommendation": "Aumentar riego y sombra."
            },
            {
                "type": "general",
                "severity": "info",
                "message": "Condiciones favorables para el cultivo de cacao.",
                "recommendation": "Mantener monitoreo regular."
            }
        ]
    }
