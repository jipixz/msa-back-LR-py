"""Funciones puras de main.py: JSON seguro, claves de cache y validación del modelo."""
import json
import math

import numpy as np
import pandas as pd


def test_dataframe_a_json_sin_nan_ni_infinitos(app_module):
    df = pd.DataFrame({"a": [1.0, np.nan, np.inf, -np.inf], "b": ["x", None, "y", "z"]})
    salida = app_module.dataframe_to_json_safe(df)
    texto = json.dumps(salida)  # no debe lanzar por NaN/Infinity
    assert "NaN" not in texto and "Infinity" not in texto


def test_clave_de_cache_determinista(app_module):
    a = app_module.get_cache_key("temperaturaBME", 24, False)
    b = app_module.get_cache_key("temperaturaBME", 24, False)
    c = app_module.get_cache_key("temperaturaBME", 48, False)
    assert a == b and a != c


def test_validacion_con_pocos_datos_devuelve_error(app_module):
    df = pd.DataFrame({"fecha": pd.date_range("2026-01-01", periods=10, freq="h"), "humedadSuelo": range(10)})
    r = app_module.validate_prediction_model(df, "humedadSuelo")
    assert "error" in r


def test_validacion_con_datos_suficientes_da_metricas_finitas(app_module):
    n = 24 * 20
    fechas = pd.date_range("2026-01-01", periods=n, freq="30min")
    valores = 45 + 3 * np.sin(np.arange(n) / 12)
    df = pd.DataFrame({"fecha": fechas, "humedadSuelo": valores})
    r = app_module.validate_prediction_model(df, "humedadSuelo")
    assert "error" not in r, r
    for k in ("mae", "rmse", "accuracy_score"):
        assert math.isfinite(r[k])
