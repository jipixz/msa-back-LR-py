"""Cloudflare Access y CORS: rutas /access/* y orígenes explícitos (nunca "*")."""
import pytest

FRONT = "https://msafront.jipixz.com"


def test_ping(client):
    r = client.get("/access/ping")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_return_regresa_al_front(client):
    r = client.get("/access/return", params={"to": f"{FRONT}/predictions?x=1"}, follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == f"{FRONT}/predictions?x=1"


@pytest.mark.parametrize("malo", [
    "https://google.com", "https://evil.com/", "https://msafront.jipixz.com.evil.com/",
    "https://msafront.jipixz.com@evil.com/", "//evil.com", "javascript:alert(1)", "",
])
def test_return_rechaza_otros_destinos(client, malo):
    r = client.get("/access/return", params={"to": malo}, follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == FRONT


def test_cors_origen_permitido_con_credenciales(client):
    r = client.get("/access/ping", headers={"Origin": FRONT})
    assert r.headers.get("access-control-allow-origin") == FRONT
    assert r.headers.get("access-control-allow-credentials") == "true"


def test_cors_origen_desconocido_sin_cabecera(client):
    r = client.get("/access/ping", headers={"Origin": "https://evil.com"})
    assert "access-control-allow-origin" not in r.headers


def test_cors_nunca_comodin(app_module):
    # Decisión: con cookies de Access el navegador rechaza "*".
    assert "*" not in app_module.CORS_ORIGINS
    assert FRONT in app_module.CORS_ORIGINS
