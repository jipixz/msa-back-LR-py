"""Pruebas estáticas y de decisiones tomadas (no ejecutan el servicio)."""
import ast
import os
from collections import Counter

import pytest

# Raíz del repo: el cwd si ahí está apps/main.py (p. ej. pruebas copiadas a otra carpeta), si no, junto a tests/
RAIZ = os.getcwd() if os.path.exists(os.path.join(os.getcwd(), "apps", "main.py")) else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _arbol():
    with open(os.path.join(RAIZ, "apps", "main.py"), encoding="utf-8") as f:
        return ast.parse(f.read())


def test_main_compila():
    _arbol()


@pytest.mark.xfail(reason="generar_alertas_recomendaciones está definida dos veces; la primera es código muerto", strict=True)
def test_sin_funciones_duplicadas():
    nombres = Counter(n.name for n in _arbol().body if isinstance(n, ast.FunctionDef))
    duplicadas = [k for k, v in nombres.items() if v > 1]
    assert duplicadas == []


def test_requirements_sin_lru_cache():
    # Decisión: lru_cache viene de functools; el paquete de PyPI se quitó de requirements.
    with open(os.path.join(RAIZ, "requirements.txt"), encoding="utf-8") as f:
        assert "lru_cache" not in f.read()


def test_pycache_no_versionado():
    with open(os.path.join(RAIZ, ".gitignore"), encoding="utf-8") as f:
        assert "__pycache__" in f.read()
