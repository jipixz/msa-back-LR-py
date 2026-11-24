"""Tiempos de respuesta de la pestaña de Predicciones y de Mongo.

Corre contra el servicio real:  PERF_BASE_URL=http://127.0.0.1:8001 pytest -m perf -s
Los umbrales son los de una Raspberry Pi 4; si un endpoint los rebasa, la prueba falla y
muestra cuánto tardó.
"""
import os
import time

import pytest

BASE = os.getenv("PERF_BASE_URL")
pytestmark = [pytest.mark.perf, pytest.mark.skipif(not BASE, reason="define PERF_BASE_URL para medir tiempos")]

# (ruta, umbral en segundos) — lo que pide el front en /predictions
PREDICCIONES = [
    ("/predictions/temperature?days=7", 3.0),
    ("/predictions/humidity", 3.0),
    ("/predictions/rainfall?days=7", 3.0),
    ("/predictions/soil-moisture?days=7", 3.0),
    ("/predictions/alerts", 3.0),
]


def _medir(cliente, ruta):
    t0 = time.perf_counter()
    r = cliente.get(ruta)
    return r, time.perf_counter() - t0


@pytest.fixture(scope="module")
def cliente():
    import httpx
    with httpx.Client(base_url=BASE, timeout=60) as c:
        yield c


@pytest.mark.parametrize("ruta,umbral", PREDICCIONES)
def test_prediccion_en_caliente(cliente, ruta, umbral):
    _medir(cliente, ruta)  # primera llamada: llena caches
    r, dt = _medir(cliente, ruta)
    print(f"{ruta}: {dt:.2f} s")
    assert r.status_code == 200
    assert dt < umbral, f"{ruta} tardó {dt:.2f} s (umbral {umbral} s)"


def test_pestana_predicciones_completa(cliente):
    """Las 5 peticiones que hace la pestaña, en serie (peor caso del navegador)."""
    total = sum(_medir(cliente, ruta)[1] for ruta, _ in PREDICCIONES)
    print(f"pestaña completa: {total:.2f} s")
    assert total < 8.0


def test_model_metrics(cliente):
    r, dt = _medir(cliente, "/model-metrics")
    print(f"/model-metrics: {dt:.2f} s")
    assert r.status_code == 200
    assert dt < 10.0, f"/model-metrics tardó {dt:.2f} s"


def test_get_datos_con_limite(cliente):
    r, dt = _medir(cliente, "/get_datos?limite=100")
    print(f"/get_datos?limite=100: {dt:.2f} s")
    assert r.status_code == 200
    assert dt < 3.0


def test_mongo_directo():
    """Conteo y últimas 300 lecturas directo en Mongo (sin pasar por FastAPI)."""
    from dotenv import load_dotenv
    load_dotenv("./datos.env")
    if not os.getenv("MONGO_DB"):
        pytest.skip("sin datos.env")
    from pymongo import MongoClient
    host, port = os.getenv("MONGO_HOST", "localhost"), int(os.getenv("MONGO_PORT", 27017))
    user, pwd = os.getenv("MONGO_USER"), os.getenv("MONGO_PASSWORD")
    uri = f"mongodb://{user}:{pwd}@{host}:{port}/{os.getenv('MONGO_AUTH_DB', 'admin')}" if user and pwd else f"mongodb://{host}:{port}"
    col = MongoClient(uri, serverSelectionTimeoutMS=5000)[os.getenv("MONGO_DB")][os.getenv("MONGO_COLLECTION")]
    t0 = time.perf_counter(); total = col.estimated_document_count(); t_count = time.perf_counter() - t0
    t0 = time.perf_counter(); docs = list(col.find().sort("fecha", -1).limit(300)); t_find = time.perf_counter() - t0
    print(f"mongo: {total} docs · conteo {t_count*1000:.0f} ms · últimas 300 {t_find*1000:.0f} ms")
    assert t_count < 1.0 and t_find < 2.0
    assert len(docs) > 0
