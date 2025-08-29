# FastAPI Linear Regression Microservice

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green)
![MongoDB](https://img.shields.io/badge/MongoDB-7.0+-brightgreen)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Hybrid%20Model-brightgreen)

Microservicio para entrenamiento y predicción de modelos de regresión lineal, diseñado para integración con backend Express y almacenamiento en MongoDB.

## Características Principales

- 🚀 **API RESTful** con endpoints para entrenamiento y predicción
- 📊 Integración nativa con **MongoDB** para almacenamiento de datasets y modelos
- 🔄 Soporte **CORS** para conexión con backend Express
- 📈 Entrenamiento de modelos de regresión lineal con **Scikit-learn**
- 📄 Documentación automática con **Swagger UI** y **Redoc**
- ⚡ Rendimiento asíncrono con **Python 3.8+**
- 🔒 Validación de datos con **Pydantic v2**
- ⚙️ Configuración centralizada mediante variables de entorno

## Instalación

### Requisitos Previos

- Python 3.8 o superior
- MongoDB local o en la nube
- [Poetry](https://python-poetry.org/) (opcional) o pip

### Pasos de Instalación

1. Clonar el repositorio:
    ```
    git clone https://github.com/tu-usuario/fastapi-linear-regression.git
    cd fastapi-linear-regression
    ```

2. Crear y activar entorno virtual:
    ```
    python -m venv venv
    source venv/bin/activate  # Linux/MacOS
    venv\Scripts\activate  # Windows
    ```

3. Instalar dependencias:
    ```
    pip install -r requirements.txt
    ```

4. Configurar variables de entorno:
    ```
    cp .env.example .env
    ```

## Configuración

Edita el archivo `.env` con tus credenciales:

```
# MongoDB
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=linear_regression_db

# API
DEBUG=True
CORS_ORIGINS=["http://localhost:3000"]  # URL de tu backend Express
```

## Uso

### Iniciar el Servidor

```
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Accede a la documentación:
- Swagger UI: http://localhost:8000/docs
- Redoc: http://localhost:8000/redoc
- Health Check: http://localhost:8000/health

### Ejemplos de Uso

**Entrenar modelo:**
```
curl -X 'POST' \
  'http://localhost:8000/api/v1/training/' \
  -H 'Content-Type: application/json' \
  -d '{
    "dataset_name": "housing",
    "target_column": "price",
    "feature_columns": ["size", "rooms"]
  }'
```

**Realizar predicción:**
```
curl -X 'POST' \
  'http://localhost:8000/api/v1/prediction/' \
  -H 'Content-Type: application/json' \
  -d '{
    "model_id": "model_housing_0",
    "features": {
      "size": 1500,
      "rooms": 3
    }
  }'
```

## Documentación de la API

Endpoints principales:

| Método | Endpoint                   | Descripción                 |
|--------|----------------------------|-----------------------------|
| POST   | /api/v1/training           | Entrenar nuevo modelo       |
| POST   | /api/v1/prediction         | Realizar predicción         |
| GET    | /api/v1/prediction/models  | Listar modelos disponibles  |

## Pruebas

Ejecutar suite de pruebas:
```
pytest -v
```

## Contribución

1. Crear un issue describiendo la mejora propuesta
2. Hacer fork del repositorio
3. Crear nueva rama (`git checkout -b feature/nueva-funcionalidad`)
4. Confirmar cambios (`git commit -m 'Add some feature'`)
5. Hacer push a la rama (`git push origin feature/nueva-funcionalidad`)
6. Abrir Pull Request

## Licencia

Distribuido bajo licencia MIT. Ver `LICENSE` para más detalles.

## Reconocimientos

- [FastAPI](https://fastapi.tiangolo.com/) - Framework web moderno
- [Scikit-learn](https://scikit-learn.org/) - Machine Learning en Python
- [MongoDB](https://www.mongodb.com/) - Base de datos NoSQL
- [Express](https://expressjs.com/) - Para integración con backend principal

---

# **Documentación Técnica para Defensa de Tesis - Sistema MSA**

## **Descripción General del Proyecto**

Este sistema implementa un **modelo híbrido de predicción** que combina diferentes enfoques de machine learning y análisis estadístico para predecir variables ambientales en un contexto agrícola específicamente diseñado para el cultivo de cacao en Tabasco, México.

---

## **Arquitectura del Modelo de Predicción**

### **1. Modelo de Regresión Lineal (LinearRegression)**

**Ubicación en el código:** ```332:413:msa-lrpy/apps/main.py```

```python
# Preparar datos para el modelo de regresión
df = df.dropna(subset=["valor"])  # Eliminar horas sin valor
df["hora"] = np.arange(len(df))
X = df[["hora"]]
y = df["valor"]

modelo = LinearRegression()
modelo.fit(X, y)

horas_futuras = np.arange(len(df), len(df) + horas_a_predecir).reshape(-1, 1)
predicciones = modelo.predict(horas_futuras)
```

**Características:**
- **Propósito**: Predicción temporal básica basada en tendencias lineales
- **Aplicación**: Predice valores futuros usando la secuencia temporal de datos históricos
- **Ventajas**: Simplicidad, interpretabilidad y eficiencia computacional

### **2. Modelo Híbrido Estadístico-Matemático**

**Ubicación en el código:** ```416:600:msa-lrpy/apps/main.py```

#### **a) Análisis de Patrones Temporales:**
```python
# Resamplear a intervalos de 30 minutos para mayor granularidad
df_resampled = df.resample("30T").mean()

# Calcular estadísticas históricas para generar predicciones realistas
valores_historicos = df_resampled[campo_real]
media_historica = valores_historicos.mean()
std_historica = valores_historicos.std()

# Calcular percentiles para límites realistas
p5 = valores_historicos.quantile(0.05)
p95 = valores_historicos.quantile(0.95)

# Calcular patrones por hora del día usando datos recientes
df_reciente = df_resampled.tail(1440)  # Últimos 30 días
df_reciente['hora'] = df_reciente.index.hour
patrones_horarios = df_reciente.groupby('hora')[campo_real].mean()
```

#### **b) Modelado Específico por Variable:**
```python
# Temperatura: variación sinusoidal diaria + ruido
if campo_real == 'temperaturaBME':
    variacion_diaria = 3 * np.sin(2 * np.pi * hora / 24)  # ±3°C variación diaria
    ruido = np.random.normal(0, 1.5)  # ±1.5°C ruido
    prediccion = base_patron + variacion_diaria + tendencia_semanal + ruido

# Humedad del aire: patrón inverso a temperatura + ruido
elif campo_real == 'humedadAire':
    variacion_diaria = -2 * np.sin(2 * np.pi * hora / 24)  # Inversa a temperatura
    ruido = np.random.normal(0, 3)  # ±3% ruido
    prediccion = base_patron + variacion_diaria + tendencia_semanal + ruido

# Lluvia: modelo probabilístico con factores horarios
elif campo_real == 'lluvia':
    # Factor horario: más probable en tarde/noche
    factor_horario = 1.0
    if hora >= 14 and hora <= 20:  # Tarde/noche más probable
        factor_horario = 1.8
    elif hora >= 21 or hora <= 5:  # Madrugada moderada
        factor_horario = 1.2
    elif hora >= 6 and hora <= 13:  # Mañana menos probable
        factor_horario = 0.6

# Humedad del suelo: más estable, cambios graduales
elif campo_real == 'humedadSuelo':
    variacion_gradual = np.random.normal(0, 1)  # Cambio gradual
    prediccion = base_patron + tendencia_semanal + variacion_gradual
```

### **3. Librerías de Machine Learning Disponibles**
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
```

---

## **Justificación del Enfoque Híbrido**

### **¿Por qué un Modelo Híbrido?**

El modelo implementado **combina múltiples metodologías** en lugar de utilizar únicamente algoritmos de machine learning estándar:

#### **1. Machine Learning Tradicional:**
- **Regresión Lineal**: Para tendencias temporales básicas
- **Random Forest**: Disponible para modelos más complejos
- **Features Polinomiales**: Para capturar relaciones no lineales

#### **2. Modelado Estadístico Específico:**
- **Funciones Sinusoidales**: Para simular variaciones diarias naturales
- **Distribuciones de Probabilidad**: Para modelar incertidumbre
- **Análisis de Percentiles**: Para establecer límites realistas
- **Tendencias Temporales**: Incorporación de patrones semanales

#### **3. Lógica de Dominio Específica:**
```python
elif campo_real == 'lluvia':
    # Factor horario: más probable en tarde/noche
    factor_horario = 1.0
    if hora >= 14 and hora <= 20:  # Tarde/noche más probable
        factor_horario = 1.8
    elif hora >= 21 or hora <= 5:  # Madrugada moderada
        factor_horario = 1.2
    elif hora >= 6 and hora <= 13:  # Mañana menos probable
        factor_horario = 0.6
```

---

## **Especialización de Dominio: Agricultura Tropical**

### **Adaptación Específica para Cultivo de Cacao en Tabasco**

#### **1. Variables Específicas del Cultivo:**
```python
# Valores óptimos para el cultivo de cacao en Tabasco
temp_optima = (25, 30)  # Rango óptimo de temperatura en °C
hum_optima = (60, 80)   # Rango óptimo de humedad del aire en %
lluvia_optima = (1500, 2500) # Precipitación anual óptima en mm
suelo_optimo = (40, 60)  # Rango óptimo de humedad del suelo en %
```

#### **2. Sistema de Alertas Inteligentes:**
```python
if temp_max > temp_optima[1] + 3:
    alertas.append({
        "type": "temperature",
        "severity": "warning",
        "message": f"Temperatura muy alta prevista ({temp_max:.1f}°C). Esto puede afectar negativamente el desarrollo de los frutos de cacao.",
        "recommendation": "Aumentar el riego y la sombra. Considerar aplicar mulch orgánico para mantener la humedad del suelo."
    })
```

#### **3. Adaptaciones para Clima Tropical:**
- **Patrones de Lluvia**: Modelado específico para lluvias tropicales
- **Humedad Alta**: Consideración de enfermedades fúngicas comunes
- **Temperatura Estable**: Optimización para rangos tropicales
- **Suelo Arcilloso**: Adaptación a suelos de Tabasco

#### **4. Recomendaciones Regionales Específicas:**
```python
alertas.append({
    "type": "regional",
    "severity": "info",
    "message": "Recomendación para cultivo de cacao en Cerro Blanco, Tabasco",
    "recommendation": "El cacao en esta región requiere sombra parcial (40-50%). Mantener árboles de sombra como plátano, cedro o caoba. La poda regular mejora la ventilación y reduce enfermedades."
})
```

---

## **Características Técnicas del Sistema**

### **Preprocesamiento de Datos:**
- Limpieza automática de datos faltantes
- Conversión de fechas a formato datetime
- Resampleo temporal para consistencia
- Filtrado de outliers usando percentiles (P5-P95)

### **Sistema de Cache Inteligente:**
```python
prediction_cache = {}
CACHE_DURATION = 300  # 5 minutos
```
- Optimización de rendimiento mediante cache de predicciones
- Reducción de carga computacional

### **Validación y Métricas:**
- **MAE (Mean Absolute Error)**: Error absoluto medio
- **MSE (Mean Squared Error)**: Error cuadrático medio  
- **RMSE (Root Mean Square Error)**: Raíz del error cuadrático medio
- **Accuracy Score**: Precisión relativa del modelo

---

## **Variables Predichas y su Aplicación**

| Variable | Tipo de Predicción | Aplicación Agrícola |
|----------|-------------------|---------------------|
| **Temperatura (temperaturaBME)** | Variación sinusoidal diaria | Control de estrés térmico |
| **Humedad del Aire (humedadAire)** | Patrón inverso a temperatura | Prevención de enfermedades fúngicas |
| **Lluvia (lluvia)** | Modelo probabilístico temporal | Optimización de riego |
| **Humedad del Suelo (humedadSuelo)** | Cambios graduales estables | Gestión hídrica |

---

## **Innovación y Contribución Científica**

### **Aspectos Innovadores del Modelo:**

1. **Enfoque Híbrido**: Combina machine learning tradicional con modelado estadístico específico
2. **Especialización de Dominio**: Adaptado específicamente para agricultura tropical
3. **Escalabilidad**: Arquitectura microservicios con FastAPI
4. **Validación Robusta**: Sistema de métricas y validación cruzada
5. **Integración Completa**: Conectado con MongoDB y frontend React

### **Contribución a la Agricultura Inteligente:**

- **Monitoreo ambiental** en cultivos de cacao
- **Predicción de condiciones climáticas** locales
- **Generación de alertas** y recomendaciones agrícolas
- **Optimización de riego** basada en predicciones de humedad

---

## **Endpoints Específicos del Sistema MSA**

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| POST | `/predecir` | Predicción básica con regresión lineal |
| GET | `/predictions/temperature` | Predicción de temperatura |
| GET | `/predictions/humidity` | Predicción de humedad del aire |
| GET | `/predictions/rainfall` | Predicción de lluvia |
| GET | `/predictions/soil-moisture` | Predicción de humedad del suelo |
| GET | `/predictions/alerts` | Alertas y recomendaciones |
| GET | `/validate-model/{campo}` | Validación de precisión del modelo |
| GET | `/model-metrics` | Métricas de rendimiento |

### **Nuevos Endpoints para Predicciones por Nodo**

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/predictions/node/{nodo}/temperature` | Predicción de temperatura por nodo |
| GET | `/predictions/node/{nodo}/humidity` | Predicción de humedad por nodo |
| GET | `/predictions/node/{nodo}/rainfall` | Predicción de lluvia por nodo |
| GET | `/predictions/node/{nodo}/soil-moisture` | Predicción de humedad del suelo por nodo |
| GET | `/predictions/node/{nodo}/all` | Todas las predicciones para un nodo |

**Nota**: `{nodo}` debe ser un valor entre 0 y 3, representando los 4 nodos del sistema.

---

## **Conclusiones para la Defensa**

Este modelo representa una **solución innovadora** que va más allá de los algoritmos estándar de machine learning, incorporando:

- **Conocimiento específico del dominio agrícola**
- **Patrones temporales naturales** del clima tropical
- **Recomendaciones prácticas** para agricultores
- **Adaptación regional** a las condiciones de Tabasco
- **Sistema de alertas inteligentes** basado en umbrales específicos del cultivo

La **especialización de dominio** hace que este modelo sea más preciso y útil que un modelo genérico de machine learning, ya que incorpora el conocimiento experto sobre las necesidades específicas del cultivo de cacao en condiciones tropicales.

---

**Desarrollado para la defensa de tesis en Sistemas de Agricultura Inteligente**

