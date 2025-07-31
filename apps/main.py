from fastapi import FastAPI, HTTPException, Body, Query
from pydantic import BaseModel
from dotenv import load_dotenv
from sshtunnel import SSHTunnelForwarder
from pymongo import MongoClient
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
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
load_dotenv("./datos.env")

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
        limite = int(os.getenv("LIMIT_DOCUMENTOS", 300))
        
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

        logger.info(f"Datos obtenidos: {len(df)} registros")
        logger.info(f"Columnas disponibles: {list(df.columns)}")

        # Mapear campo del frontend al campo real de la BD
        campo_real = nombre_campo
        
        # Filtrar solo registros con la estructura correcta
        campos_esperados = ['humedadSuelo', 'temperaturaBME', 'humedadAire', 'lluvia', 'presion', 'luminosidad']
        df_filtrado = df[df.columns.intersection(campos_esperados + ['fecha', '__v'])]
        
        if 'valor' in df.columns:
            df_filtrado = df_filtrado.drop(columns=['valor'], errors='ignore')
            logger.info("Eliminando registros con campo 'valor' (datos antiguos de prueba)")
        
        if campo_real not in df_filtrado.columns:
            logger.error(f"Columna '{campo_real}' no encontrada. Columnas disponibles: {list(df_filtrado.columns)}")
            raise HTTPException(status_code=500, detail=f"Columna '{campo_real}' no encontrada.")
        
        df = df_filtrado
        
        if "fecha" not in df.columns:
            raise HTTPException(status_code=500, detail="Columna 'fecha' no encontrada.")

        # Convertir fecha a datetime y limpiar datos
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df = df.dropna(subset=["fecha", campo_real])
        
        if len(df) < 50:
            raise HTTPException(status_code=500, detail="Datos insuficientes para predicción (mínimo 50 registros).")

        # Ordenar por fecha
        df = df.sort_values('fecha')
        df = df.set_index("fecha")
        
        # Resamplear a intervalos de 30 minutos para mayor granularidad
        df_resampled = df.resample("30T").mean()
        df_resampled = df_resampled.dropna(subset=[campo_real])
        
        if len(df_resampled) < 20:
            raise HTTPException(status_code=500, detail="Datos insuficientes después del resample.")
        
        logger.info(f"Datos después de resample: {len(df_resampled)} registros")
        
        # Calcular estadísticas históricas para generar predicciones realistas
        valores_historicos = df_resampled[campo_real]
        media_historica = valores_historicos.mean()
        std_historica = valores_historicos.std()
        
        # Calcular percentiles para límites realistas
        p5 = valores_historicos.quantile(0.05)
        p95 = valores_historicos.quantile(0.95)
        
        # Calcular patrones por hora del día usando datos recientes (últimos 30 días)
        df_reciente = df_resampled.tail(1440)  # Últimos 30 días aprox (30*24*2 = 1440 registros de 30min)
        df_reciente['hora'] = df_reciente.index.hour
        patrones_horarios = df_reciente.groupby('hora')[campo_real].mean()
        
        logger.info(f"Estadísticas - Media: {media_historica:.2f}, Std: {std_historica:.2f}, Rango: [{p5:.2f}, {p95:.2f}]")
        
        # Generar predicciones para 7 días (4 puntos por día: 6:00, 12:00, 18:00, 24:00)
        ultima_fecha = df_resampled.index[-1]
        resultados = []
        
        for dia in range(7):
            fecha_dia = ultima_fecha + pd.Timedelta(days=dia+1)
            
            # Generar 4 predicciones por día
            for hora in [6, 12, 18, 24]:
                if hora == 24:
                    hora = 0
                    fecha_pred = fecha_dia + pd.Timedelta(days=1)
                else:
                    fecha_pred = fecha_dia.replace(hour=hora, minute=0, second=0, microsecond=0)
                
                # Base de la predicción: patrón horario + tendencia semanal ligera
                base_patron = patrones_horarios.get(hora, media_historica)
                
                # Tendencia semanal muy ligera (±2% por semana)
                tendencia_semanal = np.random.uniform(-0.02, 0.02) * base_patron * (dia / 7)
                
                # Variación natural específica por tipo de sensor
                if campo_real == 'temperaturaBME':
                    # Temperatura: variación sinusoidal diaria + ruido
                    variacion_diaria = 3 * np.sin(2 * np.pi * hora / 24)  # ±3°C variación diaria
                    ruido = np.random.normal(0, 1.5)  # ±1.5°C ruido
                    prediccion = base_patron + variacion_diaria + tendencia_semanal + ruido
                    
                elif campo_real == 'humedadAire':
                    # Humedad del aire: patrón inverso a temperatura + ruido
                    variacion_diaria = -2 * np.sin(2 * np.pi * hora / 24)  # Inversa a temperatura
                    ruido = np.random.normal(0, 3)  # ±3% ruido
                    prediccion = base_patron + variacion_diaria + tendencia_semanal + ruido
                    
                elif campo_real == 'lluvia':
                    # Lluvia: lógica especial que no depende tanto de patrones históricos
                    # ya que la lluvia puede ser muy esporádica en los datos históricos
                    
                    # Factor horario: más probable en tarde/noche
                    factor_horario = 1.0
                    if hora >= 14 and hora <= 20:  # Tarde/noche más probable
                        factor_horario = 1.8
                    elif hora >= 21 or hora <= 5:  # Madrugada moderada
                        factor_horario = 1.2
                    elif hora >= 6 and hora <= 13:  # Mañana menos probable
                        factor_horario = 0.6
                    
                    # Usar base histórica solo si es significativa
                    base_lluvia = max(0, base_patron) if base_patron > 0.5 else 0
                    
                    # Generar predicción realista independiente de límites históricos
                    probabilidad_lluvia = np.random.random()
                    
                    if base_lluvia > 1.0:  # Si hay patrón histórico significativo
                        # Usar patrón histórico con variación
                        prediccion = base_lluvia * factor_horario * np.random.uniform(0.5, 2.0)
                        # Añadir posibilidad de picos de lluvia
                        if probabilidad_lluvia < 0.15:  # 15% chance de lluvia intensa
                            prediccion += np.random.uniform(5, 15)
                    else:
                        # Sin patrón histórico fuerte, usar modelo probabilístico
                        if probabilidad_lluvia < 0.25:  # 25% probabilidad de lluvia
                            if probabilidad_lluvia < 0.05:  # 5% lluvia intensa
                                prediccion = np.random.uniform(8, 25) * factor_horario
                            elif probabilidad_lluvia < 0.15:  # 10% lluvia moderada
                                prediccion = np.random.uniform(3, 8) * factor_horario
                            else:  # 10% lluvia ligera
                                prediccion = np.random.uniform(0.5, 3) * factor_horario
                        else:
                            # Sin lluvia o lluvia mínima
                            prediccion = np.random.uniform(0, 0.3)
                    
                    # Aplicar tendencia semanal solo si es positiva
                    if tendencia_semanal > 0:
                        prediccion += tendencia_semanal
                    
                    # Asegurar valores realistas para lluvia (no aplicar límites p5/p95)
                    prediccion = max(0, min(50, prediccion))  # Límite máximo de 50mm por predicción
                    
                elif campo_real == 'humedadSuelo':
                    # Humedad del suelo: más estable, cambios graduales
                    variacion_gradual = np.random.normal(0, 1)  # Cambio gradual
                    prediccion = base_patron + tendencia_semanal + variacion_gradual
                    
                else:
                    # Para otros campos: variación estándar
                    ruido = np.random.normal(0, std_historica * 0.1)
                    prediccion = base_patron + tendencia_semanal + ruido
                
                # Aplicar límites realistas basados en datos históricos (EXCEPTO para lluvia)
                if campo_real != 'lluvia':
                    prediccion = max(p5, min(p95, prediccion))
                
                # Asegurar valores positivos para ciertos campos
                if campo_real in ['lluvia', 'humedadSuelo', 'humedadAire']:
                    prediccion = max(0, prediccion)
                
                # Formatear día para visualización
                if dia == 0:
                    dia_label = "Hoy"
                elif dia == 1:
                    dia_label = "Mañana"
                else:
                    dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
                    dia_label = dias_semana[fecha_pred.weekday()]
                
                resultados.append({
                    "predicted": round(float(prediccion), 2),
                    "day": fecha_pred.strftime('%Y-%m-%d'),
                    "day_label": dia_label,
                    "hour": hora if hora != 0 else 24,
                    "fecha_completa": fecha_pred.strftime('%Y-%m-%d %H:%M:%S'),
                    "actual": None,
                    "probability": None
                })
        
        logger.info(f"Generadas {len(resultados)} predicciones para {nombre_campo}")
        
        respuesta = {
            "predictions": resultados,
            "field": campo_real,
            "total_predictions": len(resultados),
            "historical_stats": {
                "mean": round(float(media_historica), 2),
                "std": round(float(std_historica), 2),
                "min": round(float(p5), 2),
                "max": round(float(p95), 2),
                "total_records": len(df_resampled)
            }
        }
        
        # Generar gráfica si se solicita
        if graficar:
            try:
                plt.figure(figsize=(12, 6))
                
                # Datos históricos (últimos 7 días)
                datos_recientes = df_resampled.tail(336)  # 7 días * 24 horas * 2 (cada 30min) = 336
                plt.plot(datos_recientes.index, datos_recientes[campo_real], 
                        label="Datos históricos (últimos 7 días)", color='blue', alpha=0.7)
                
                # Predicciones
                fechas_pred = [pd.to_datetime(r['fecha_completa']) for r in resultados]
                valores_pred = [r['predicted'] for r in resultados]
                plt.plot(fechas_pred, valores_pred, 
                        label="Predicciones (7 días)", color='red', marker='o', linestyle='--')
                
                plt.title(f"Predicción de {campo_real} - Próximos 7 días")
                plt.xlabel("Fecha")
                plt.ylabel(f"{campo_real}")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                buffer = io.BytesIO()
                plt.savefig(buffer, format="png", dpi=150, bbox_inches='tight')
                buffer.seek(0)
                imagen_base64 = base64.b64encode(buffer.read()).decode("utf-8")
                buffer.close()
                plt.close()
                
                respuesta["grafica"] = imagen_base64
                
            except Exception as e:
                logger.warning(f"Error al generar gráfica: {e}")
        
        return respuesta
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en predicción de {nombre_campo}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")
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

# Función para generar alertas y recomendaciones basadas en predicciones
def generar_alertas_recomendaciones():
    try:
        logger.info("Generando alertas y recomendaciones basadas en predicciones")
        
        # Obtener predicciones para cada variable
        temp_data = predecir_campo("temperaturaBME", 168, False)
        hum_data = predecir_campo("humedadAire", 168, False)
        lluvia_data = predecir_campo("lluvia", 168, False)
        suelo_data = predecir_campo("humedadSuelo", 168, False)
        
        alertas = []
        
        # Valores óptimos para el cultivo de cacao en Tabasco
        temp_optima = (25, 30)  # Rango óptimo de temperatura en °C
        hum_optima = (60, 80)   # Rango óptimo de humedad del aire en %
        lluvia_optima = (1500, 2500) # Precipitación anual óptima en mm (aprox. 4-7mm diarios)
        suelo_optimo = (40, 60)  # Rango óptimo de humedad del suelo en %
        
        # Analizar temperatura
        if temp_data and "predictions" in temp_data and temp_data["predictions"]:
            # Obtener promedio de temperatura para los próximos 7 días
            temps = [p.get("predicted", 0) for p in temp_data["predictions"] if p.get("predicted") is not None]
            if temps:
                temp_promedio = sum(temps) / len(temps)
                temp_max = max(temps)
                
                if temp_max > temp_optima[1] + 3:
                    # Alerta por temperatura muy alta
                    alertas.append({
                        "type": "temperature",
                        "severity": "warning",
                        "message": f"Temperatura muy alta prevista ({temp_max:.1f}°C). Esto puede afectar negativamente el desarrollo de los frutos de cacao.",
                        "recommendation": "Aumentar el riego y la sombra. Considerar aplicar mulch orgánico para mantener la humedad del suelo."
                    })
                elif temp_max > temp_optima[1]:
                    # Alerta por temperatura alta
                    alertas.append({
                        "type": "temperature",
                        "severity": "warning",
                        "message": f"Temperatura ligeramente alta prevista ({temp_max:.1f}°C).",
                        "recommendation": "Monitorear el riego y asegurar sombra adecuada para las plantas de cacao."
                    })
                elif temp_promedio < temp_optima[0]:
                    # Alerta por temperatura baja
                    alertas.append({
                        "type": "temperature",
                        "severity": "warning",
                        "message": f"Temperatura promedio baja prevista ({temp_promedio:.1f}°C).",
                        "recommendation": "Asegurar que las plantas tengan protección contra vientos fríos. Reducir la poda durante este período."
                    })
                else:
                    # Temperatura óptima
                    alertas.append({
                        "type": "temperature",
                        "severity": "info",
                        "message": f"Temperatura dentro del rango óptimo para el cacao ({temp_promedio:.1f}°C).",
                        "recommendation": "Mantener las prácticas actuales de manejo de sombra y riego."
                    })
        
        # Analizar humedad del aire
        if hum_data and "predictions" in hum_data and hum_data["predictions"]:
            # Obtener promedio de humedad para los próximos 7 días
            hums = [p.get("predicted", 0) for p in hum_data["predictions"] if p.get("predicted") is not None]
            if hums:
                hum_promedio = sum(hums) / len(hums)
                
                if hum_promedio > hum_optima[1] + 10:
                    # Alerta por humedad muy alta
                    alertas.append({
                        "type": "humidity",
                        "severity": "warning",
                        "message": f"Humedad del aire muy alta prevista ({hum_promedio:.0f}%). Riesgo elevado de enfermedades fúngicas.",
                        "recommendation": "Aumentar la ventilación entre plantas. Considerar aplicación preventiva de fungicidas orgánicos. Reducir el riego si es posible."
                    })
                elif hum_promedio > hum_optima[1]:
                    # Alerta por humedad alta
                    alertas.append({
                        "type": "humidity",
                        "severity": "warning",
                        "message": f"Humedad del aire ligeramente alta prevista ({hum_promedio:.0f}%).",
                        "recommendation": "Monitorear signos de enfermedades fúngicas. Asegurar buena circulación de aire entre plantas."
                    })
                elif hum_promedio < hum_optima[0] - 10:
                    # Alerta por humedad muy baja
                    alertas.append({
                        "type": "humidity",
                        "severity": "warning",
                        "message": f"Humedad del aire muy baja prevista ({hum_promedio:.0f}%). Riesgo de estrés hídrico.",
                        "recommendation": "Aumentar frecuencia de riego. Aplicar mulch para conservar humedad. Considerar riego por aspersión en horas tempranas."
                    })
                elif hum_promedio < hum_optima[0]:
                    # Alerta por humedad baja
                    alertas.append({
                        "type": "humidity",
                        "severity": "warning",
                        "message": f"Humedad del aire ligeramente baja prevista ({hum_promedio:.0f}%).",
                        "recommendation": "Aumentar ligeramente el riego. Monitorear el estado de las hojas."
                    })
                else:
                    # Humedad óptima
                    alertas.append({
                        "type": "humidity",
                        "severity": "info",
                        "message": f"Humedad del aire dentro del rango óptimo para el cacao ({hum_promedio:.0f}%).",
                        "recommendation": "Mantener las prácticas actuales de manejo."
                    })
        
        # Analizar lluvia
        if lluvia_data and "predictions" in lluvia_data and lluvia_data["predictions"]:
            # Sumar precipitación prevista para los próximos 7 días
            lluvias = [p.get("predicted", 0) for p in lluvia_data["predictions"] if p.get("predicted") is not None]
            if lluvias:
                lluvia_total = sum(lluvias)
                lluvia_diaria = lluvia_total / 7  # Promedio diario
                
                # Convertir a estimación mensual para comparar con rangos óptimos
                lluvia_mensual_est = lluvia_diaria * 30
                
                if lluvia_mensual_est < 100:  # Menos de 100mm al mes es muy poco
                    alertas.append({
                        "type": "rainfall",
                        "severity": "warning",
                        "message": f"Precipitación muy baja prevista ({lluvia_total:.1f}mm en 7 días). Riesgo de sequía.",
                        "recommendation": "Implementar riego suplementario. Aplicar mulch para conservar humedad. Considerar sombra adicional temporal."
                    })
                elif lluvia_mensual_est > 250:  # Más de 250mm al mes puede ser excesivo
                    alertas.append({
                        "type": "rainfall",
                        "severity": "warning",
                        "message": f"Precipitación muy alta prevista ({lluvia_total:.1f}mm en 7 días). Riesgo de encharcamiento y enfermedades.",
                        "recommendation": "Verificar drenaje de la parcela. Monitorear signos de enfermedades fúngicas. Reducir riego suplementario."
                    })
                else:
                    alertas.append({
                        "type": "rainfall",
                        "severity": "info",
                        "message": f"Precipitación dentro del rango adecuado ({lluvia_total:.1f}mm en 7 días).",
                        "recommendation": "Mantener monitoreo regular de humedad del suelo."
                    })
        
        # Analizar humedad del suelo
        if suelo_data and "predictions" in suelo_data and suelo_data["predictions"]:
            # Obtener promedio de humedad del suelo para los próximos 7 días
            suelos = [p.get("predicted", 0) for p in suelo_data["predictions"] if p.get("predicted") is not None]
            if suelos:
                suelo_promedio = sum(suelos) / len(suelos)
                
                if suelo_promedio > suelo_optimo[1] + 10:
                    # Alerta por suelo muy húmedo
                    alertas.append({
                        "type": "soil",
                        "severity": "warning",
                        "message": f"Humedad del suelo muy alta prevista ({suelo_promedio:.0f}%). Riesgo de asfixia radicular y pudrición.",
                        "recommendation": "Mejorar drenaje. Reducir o suspender riego. Monitorear signos de pudrición en raíces y tronco."
                    })
                elif suelo_promedio > suelo_optimo[1]:
                    # Alerta por suelo húmedo
                    alertas.append({
                        "type": "soil",
                        "severity": "warning",
                        "message": f"Humedad del suelo ligeramente alta prevista ({suelo_promedio:.0f}%).",
                        "recommendation": "Reducir frecuencia de riego. Monitorear drenaje de la parcela."
                    })
                elif suelo_promedio < suelo_optimo[0] - 10:
                    # Alerta por suelo muy seco
                    alertas.append({
                        "type": "soil",
                        "severity": "warning",
                        "message": f"Humedad del suelo muy baja prevista ({suelo_promedio:.0f}%). Riesgo de estrés hídrico severo.",
                        "recommendation": "Aumentar urgentemente el riego. Aplicar mulch orgánico. Considerar riego profundo para estimular raíces."
                    })
                elif suelo_promedio < suelo_optimo[0]:
                    # Alerta por suelo seco
                    alertas.append({
                        "type": "soil",
                        "severity": "warning",
                        "message": f"Humedad del suelo ligeramente baja prevista ({suelo_promedio:.0f}%).",
                        "recommendation": "Aumentar frecuencia de riego. Aplicar mulch para conservar humedad."
                    })
                else:
                    # Humedad de suelo óptima
                    alertas.append({
                        "type": "soil",
                        "severity": "info",
                        "message": f"Humedad del suelo dentro del rango óptimo para el cacao ({suelo_promedio:.0f}%).",
                        "recommendation": "Mantener las prácticas actuales de riego."
                    })
        
        # Recomendaciones específicas para el cultivo de cacao en Cerro Blanco, Tabasco
        alertas.append({
            "type": "regional",
            "severity": "info",
            "message": "Recomendación para cultivo de cacao en Cerro Blanco, Tabasco",
            "recommendation": "El cacao en esta región requiere sombra parcial (40-50%). Mantener árboles de sombra como plátano, cedro o caoba. La poda regular mejora la ventilación y reduce enfermedades."
        })
        
        # Si no hay alertas específicas, agregar una general
        if len(alertas) <= 1:
            alertas.append({
                "type": "general",
                "severity": "info",
                "message": "Condiciones generales favorables para el cultivo de cacao.",
                "recommendation": "Mantener monitoreo regular de las condiciones ambientales y estado de las plantas."
            })
            
        return {"alerts": alertas}
        
    except Exception as e:
        logger.error(f"Error generando alertas: {e}")
        logger.error(traceback.format_exc())
        # Devolver alertas genéricas en caso de error
        return {
            "alerts": [
                {
                    "type": "general",
                    "severity": "info",
                    "message": "Monitoreo continuo recomendado.",
                    "recommendation": "Mantener vigilancia de las condiciones climáticas y estado de las plantas."
                }
            ]
        }

# Función para generar alertas y recomendaciones basadas en predicciones
def generar_alertas_recomendaciones(temp_pred=None, humedad_aire_pred=None, lluvia_pred=None, humedad_suelo_pred=None):
    """
    Genera alertas y recomendaciones basadas en las predicciones de temperatura, humedad del aire,
    lluvia y humedad del suelo para el cultivo de cacao en Cerro Blanco, Tabasco.
    
    Args:
        temp_pred: Predicciones de temperatura
        humedad_aire_pred: Predicciones de humedad del aire
        lluvia_pred: Predicciones de lluvia
        humedad_suelo_pred: Predicciones de humedad del suelo
        
    Returns:
        Diccionario con alertas y recomendaciones
    """
    try:
        alertas = []
        
        # Valores óptimos para cacao en Tabasco
        temp_optima_min, temp_optima_max = 25, 30
        humedad_aire_optima_min, humedad_aire_optima_max = 60, 80
        lluvia_optima_min, lluvia_optima_max = 2, 15  # mm/día
        humedad_suelo_optima_min, humedad_suelo_optima_max = 40, 60  # %
        
        # Verificar temperatura
        if temp_pred and len(temp_pred.get("predictions", [])) > 0:
            temp_values = [p.get("predicted") for p in temp_pred["predictions"] if p.get("predicted") is not None]
            if temp_values:
                temp_max = max(temp_values)
                temp_min = min(temp_values)
                temp_avg = sum(temp_values) / len(temp_values)
                
                if temp_max > 35:
                    alertas.append({
                        "type": "warning",
                        "title": "Temperatura Alta",
                        "message": f"La temperatura alcanzará hasta {temp_max:.1f}°C en los próximos días, lo que puede afectar el desarrollo del cacao.",
                        "recommendations": [
                            "Aumentar frecuencia de riego para mantener la humedad",
                            "Proporcionar sombra adicional a las plantas jóvenes",
                            "Aplicar mulch orgánico para mantener la humedad del suelo",
                            "Regar preferentemente en las horas más frescas (temprano en la mañana o al atardecer)"
                        ]
                    })
                elif temp_min < 20:
                    alertas.append({
                        "type": "warning",
                        "title": "Temperatura Baja",
                        "message": f"La temperatura descenderá hasta {temp_min:.1f}°C, lo que puede ralentizar el crecimiento del cacao.",
                        "recommendations": [
                            "Monitorear las plantas jóvenes que son más susceptibles",
                            "Considerar el uso de coberturas para proteger las plantas",
                            "Evitar riego excesivo durante estos días"
                        ]
                    })
                elif temp_optima_min <= temp_avg <= temp_optima_max:
                    alertas.append({
                        "type": "favorable",
                        "title": "Temperatura Óptima",
                        "message": f"La temperatura promedio de {temp_avg:.1f}°C es ideal para el desarrollo del cacao.",
                        "recommendations": [
                            "Mantener el régimen de riego regular",
                            "Aprovechar estas condiciones para actividades de poda o injerto",
                            "Monitorear el desarrollo de frutos"
                        ]
                    })
        
        # Verificar humedad del aire
        if humedad_aire_pred and len(humedad_aire_pred.get("predictions", [])) > 0:
            humedad_values = [p.get("predicted") for p in humedad_aire_pred["predictions"] if p.get("predicted") is not None]
            if humedad_values:
                humedad_max = max(humedad_values)
                humedad_min = min(humedad_values)
                humedad_avg = sum(humedad_values) / len(humedad_values)
                
                if humedad_min < 50:
                    alertas.append({
                        "type": "warning",
                        "title": "Humedad Baja",
                        "message": f"La humedad del aire descenderá hasta {humedad_min:.1f}%, lo que puede estresar las plantas de cacao.",
                        "recommendations": [
                            "Aumentar la frecuencia de riego",
                            "Aplicar riego por aspersión en las horas más calurosas",
                            "Mantener cobertura vegetal para conservar la humedad"
                        ]
                    })
                elif humedad_max > 90:
                    alertas.append({
                        "type": "warning",
                        "title": "Humedad Alta",
                        "message": f"La humedad del aire alcanzará hasta {humedad_max:.1f}%, lo que puede favorecer enfermedades fúngicas.",
                        "recommendations": [
                            "Vigilar la aparición de moniliasis y phytophthora",
                            "Asegurar buena ventilación entre plantas",
                            "Considerar aplicación preventiva de fungicidas orgánicos",
                            "Realizar podas sanitarias si es necesario"
                        ]
                    })
                elif humedad_aire_optima_min <= humedad_avg <= humedad_aire_optima_max:
                    alertas.append({
                        "type": "favorable",
                        "title": "Humedad Óptima",
                        "message": f"La humedad promedio del aire de {humedad_avg:.1f}% es ideal para el cacao.",
                        "recommendations": [
                            "Mantener prácticas regulares de manejo",
                            "Monitorear el desarrollo de flores y frutos",
                            "Buen momento para realizar polinización manual si es necesario"
                        ]
                    })
        
        # Verificar lluvia
        if lluvia_pred and len(lluvia_pred.get("predictions", [])) > 0:
            lluvia_values = [p.get("predicted") for p in lluvia_pred["predictions"] if p.get("predicted") is not None]
            if lluvia_values:
                lluvia_max = max(lluvia_values)
                lluvia_total = sum(lluvia_values)
                dias_lluvia = sum(1 for v in lluvia_values if v > 1.0)
                
                if lluvia_max > 20:
                    alertas.append({
                        "type": "warning",
                        "title": "Lluvia Intensa",
                        "message": f"Se esperan precipitaciones de hasta {lluvia_max:.1f}mm, lo que puede causar encharcamiento.",
                        "recommendations": [
                            "Verificar el drenaje de las parcelas",
                            "Evitar aplicación de fertilizantes durante estos días",
                            "Vigilar posibles deslaves en terrenos inclinados",
                            "Posponer actividades de poda o cosecha"
                        ]
                    })
                elif lluvia_total < 5 and len(lluvia_values) >= 3:
                    alertas.append({
                        "type": "warning",
                        "title": "Precipitación Insuficiente",
                        "message": f"Se esperan solo {lluvia_total:.1f}mm de lluvia en los próximos días, lo que puede causar estrés hídrico.",
                        "recommendations": [
                            "Implementar riego suplementario",
                            "Priorizar el riego en plantas jóvenes",
                            "Aplicar mulch para conservar la humedad del suelo",
                            "Regar preferentemente en las horas más frescas"
                        ]
                    })
                elif dias_lluvia >= 2 and lluvia_total >= lluvia_optima_min * dias_lluvia and lluvia_total <= lluvia_optima_max * dias_lluvia:
                    alertas.append({
                        "type": "favorable",
                        "title": "Precipitación Favorable",
                        "message": f"Se esperan {lluvia_total:.1f}mm de lluvia bien distribuidos, ideal para el cacao.",
                        "recommendations": [
                            "Aprovechar para aplicar fertilizantes orgánicos",
                            "Monitorear el desarrollo de frutos",
                            "Buen momento para realizar injertos si es necesario"
                        ]
                    })
        
        # Verificar humedad del suelo
        if humedad_suelo_pred and len(humedad_suelo_pred.get("predictions", [])) > 0:
            humedad_suelo_values = [p.get("predicted") for p in humedad_suelo_pred["predictions"] if p.get("predicted") is not None]
            if humedad_suelo_values:
                humedad_suelo_min = min(humedad_suelo_values)
                humedad_suelo_max = max(humedad_suelo_values)
                humedad_suelo_avg = sum(humedad_suelo_values) / len(humedad_suelo_values)
                
                if humedad_suelo_min < 30:
                    alertas.append({
                        "type": "warning",
                        "title": "Suelo Seco",
                        "message": f"La humedad del suelo descenderá hasta {humedad_suelo_min:.1f}%, lo que puede afectar la absorción de nutrientes.",
                        "recommendations": [
                            "Aumentar frecuencia y cantidad de riego",
                            "Aplicar mulch orgánico para retener humedad",
                            "Evitar labores que disturben el suelo",
                            "Considerar riego por goteo para optimizar el uso del agua"
                        ]
                    })
                elif humedad_suelo_max > 70:
                    alertas.append({
                        "type": "warning",
                        "title": "Suelo Saturado",
                        "message": f"La humedad del suelo alcanzará hasta {humedad_suelo_max:.1f}%, lo que puede causar problemas de aireación en las raíces.",
                        "recommendations": [
                            "Verificar y mejorar el drenaje de las parcelas",
                            "Evitar riego adicional hasta que el suelo drene",
                            "Vigilar la aparición de enfermedades radiculares",
                            "Evitar el tránsito de personas y equipos en la parcela"
                        ]
                    })
                elif humedad_suelo_optima_min <= humedad_suelo_avg <= humedad_suelo_optima_max:
                    alertas.append({
                        "type": "favorable",
                        "title": "Humedad de Suelo Óptima",
                        "message": f"La humedad promedio del suelo de {humedad_suelo_avg:.1f}% es ideal para el desarrollo radicular del cacao.",
                        "recommendations": [
                            "Mantener el régimen de riego actual",
                            "Buen momento para aplicación de biofertilizantes",
                            "Monitorear el desarrollo vegetativo de las plantas"
                        ]
                    })
        
        # Si no hay alertas específicas, agregar una recomendación general
        if not alertas:
            alertas.append({
                "type": "info",
                "title": "Monitoreo Regular",
                "message": "No se detectan condiciones extremas. Continúe con el manejo regular de su cultivo.",
                "recommendations": [
                    "Mantener observación regular de las plantas",
                    "Seguir con las prácticas habituales de manejo",
                    "Revisar el estado fitosanitario del cultivo"
                ]
            })
        
        # Agregar recomendación específica para Cerro Blanco, Tabasco
        alertas.append({
            "type": "info",
            "title": "Recomendación Regional: Cerro Blanco, Tabasco",
            "message": "Recomendaciones específicas para productores de cacao en Cerro Blanco, 5ta Sección, Tapijulapa.",
            "recommendations": [
                "Mantener sombra adecuada con especies nativas como el Samán y Cedro",
                "Implementar barreras vivas en terrenos con pendiente para evitar erosión",
                "Considerar sistemas agroforestales que combinan cacao con especies maderables y frutales",
                "Participar en las capacitaciones del programa de mejoramiento de cacao de Tabasco",
                "Aprovechar la cercanía al río para sistemas de riego en época seca"
            ]
        })
        
        return {"alerts": alertas}
    
    except Exception as e:
        logger.error(f"Error generando alertas: {e}")
        logger.error(traceback.format_exc())
        # Devolver al menos una alerta genérica en caso de error
        return {
            "alerts": [{
                "type": "info",
                "title": "Información de Cultivo",
                "message": "Mantenga un monitoreo regular de su cultivo de cacao.",
                "recommendations": [
                    "Verificar regularmente la humedad del suelo",
                    "Mantener prácticas adecuadas de manejo"
                ]
            }]
        }

# Ruta para obtener alertas y recomendaciones
@app.get("/predictions/alerts")
@app.post("/predictions/alerts")
async def get_alerts():
    try:
        # Obtener predicciones recientes para generar alertas contextuales
        temp_pred = None
        humedad_aire_pred = None
        lluvia_pred = None
        humedad_suelo_pred = None
        
        try:
            # Intentar obtener predicciones de temperatura
            temp_pred = predecir_campo("temperaturaBME", 72)  # 3 días
        except Exception as e:
            logger.warning(f"No se pudieron obtener predicciones de temperatura: {e}")
        
        try:
            # Intentar obtener predicciones de humedad del aire
            humedad_aire_pred = predecir_campo("humedadAire", 72)  # 3 días
        except Exception as e:
            logger.warning(f"No se pudieron obtener predicciones de humedad del aire: {e}")
        
        try:
            # Intentar obtener predicciones de lluvia
            lluvia_pred = predecir_campo("lluvia", 72)  # 3 días
        except Exception as e:
            logger.warning(f"No se pudieron obtener predicciones de lluvia: {e}")
        
        try:
            # Intentar obtener predicciones de humedad del suelo
            humedad_suelo_pred = predecir_campo("humedadSuelo", 72)  # 3 días
        except Exception as e:
            logger.warning(f"No se pudieron obtener predicciones de humedad del suelo: {e}")
        
        # Generar alertas basadas en las predicciones específicas para Cerro Blanco, Tabasco
        return generar_alertas_recomendaciones(
            temp_pred=temp_pred,
            humedad_aire_pred=humedad_aire_pred,
            lluvia_pred=lluvia_pred,
            humedad_suelo_pred=humedad_suelo_pred
        )
    
    except Exception as e:
        logger.error(f"Error en endpoint de alertas: {e}")
        logger.error(traceback.format_exc())
        return {
            "alerts": [{
                "type": "info",
                "title": "Información General",
                "message": "Sistema de alertas en mantenimiento. Por favor, revise más tarde.",
                "recommendations": [
                    "Continuar con las prácticas habituales de manejo"
                ]
            }]
        }
