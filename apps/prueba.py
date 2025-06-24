from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
from bson.json_util import dumps  # para convertir a JSON compatible

app = FastAPI()

client = MongoClient('mongodb://localhost:27017/')
db = client['humedad-cacao']
collection = db['humedads']

@app.get("/datos")
def obtener_datos():
    try:
        filter = {}
        cursor = collection.find(filter)
        datos = list(cursor)
        if not datos:
            raise HTTPException(status_code=404, detail="No se encontraron datos.")
        # Usamos dumps para convertir el resultado BSON a JSON string
        return dumps(datos)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error conectando a MongoDB: {e}")
