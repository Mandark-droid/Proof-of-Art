import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from pramaan_v2.server import app

    return TestClient(app)


def test_openenv_contract_routes_are_present(client):
    from pramaan_v2.server import app

    paths = {getattr(route, "path", None) for route in app.routes}
    for path in ("/reset", "/step", "/state", "/schema", "/health", "/ws", "/web/reset", "/web/step"):
        assert path in paths, path


def test_root_is_the_v2_site_not_openenv(client):
    from pramaan_v2.server import app

    flat = []
    for route in app.routes:
        flat.extend(getattr(getattr(route, "original_router", None), "routes", None) or [route])
    roots = [route for route in flat if getattr(route, "path", None) == "/"
             and "GET" in (getattr(route, "methods", None) or ())]
    assert len(roots) == 1
    assert roots[0].endpoint.__module__ == "pramaan_v2.site.routes"


def test_short_routes_redirect_to_the_apps(client):
    assert client.get("/try", follow_redirects=False).headers["location"] == "/studio/"
    assert client.get("/submit", follow_redirects=False).headers["location"] == "/studio/#/submit"
    response = client.get("/playground?task_id=tn_kolam-easy-22", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/web/?task_id=tn_kolam-easy-22"
    assert client.get("/web", follow_redirects=False).headers["location"] == "/web/"


def test_web_reset_and_step_still_work(client):
    reset = client.post("/web/reset", json={})
    assert reset.status_code == 200


def test_studio_is_mounted(client):
    response = client.get("/studio/")
    assert response.status_code == 200
    assert "gradio" in response.text.lower()


def test_root_sign_in_routes_forward_into_the_studio(client):
    response = client.get("/login/huggingface?x=1", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/studio/login/huggingface?x=1"
    assert client.get("/login/callback", follow_redirects=False).headers["location"] == "/studio/login/callback"
    assert client.get("/logout", follow_redirects=False).headers["location"] == "/studio/logout"
