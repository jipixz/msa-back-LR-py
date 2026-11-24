import os
import sys

import pytest

# Raíz del repo: el cwd si ahí está apps/main.py (p. ej. pruebas copiadas a otra carpeta), si no, junto a tests/
RAIZ = os.getcwd() if os.path.exists(os.path.join(os.getcwd(), "apps", "main.py")) else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)


@pytest.fixture(scope="session")
def app_module():
    # main.py hace load_dotenv("./datos.env") relativo al cwd: correr pytest desde la raíz del repo.
    from apps import main
    return main


@pytest.fixture(scope="session")
def client(app_module):
    from fastapi.testclient import TestClient
    return TestClient(app_module.app)
