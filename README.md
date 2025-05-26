# FastAPI Linear Regression Microservice

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green)
![MongoDB](https://img.shields.io/badge/MongoDB-7.0+-brightgreen)

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
```

