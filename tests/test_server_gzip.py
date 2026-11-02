from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.testclient import TestClient


def _application_with_compression_probe_routes(monkeypatch):
    import sys

    import scripts.check_app_configuration as check_app
    import gradio.routes

    check_app.install_windows_gradio_lock_shim()
    monkeypatch.setenv("HF_TOKEN", "")
    monkeypatch.setenv("SYSTEM", "spaces")
    monkeypatch.setenv("SPACE_ID", "pramaan-gzip-test")
    monkeypatch.setenv("OAUTH_CLIENT_ID", "pramaan-gzip-test")
    monkeypatch.setattr(gradio.routes, "attach_oauth", lambda _app: None)
    from pramaan_env.server.app import app

    if not any(getattr(route, "path", None) == "/__gzip_probe/html" for route in app.routes):
        async def large_html():
            return HTMLResponse("<main>" + ("Pramaan benchmark " * 1024) + "</main>")

        async def event_stream():
            async def events():
                for index in range(3):
                    yield f"data: update-{index}\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")

        app.add_api_route("/__gzip_probe/html", large_html, methods=["GET"])
        app.add_api_route("/__gzip_probe/events", event_stream, methods=["GET"])
    return app


def test_real_server_app_gzips_large_html(monkeypatch):
    app = _application_with_compression_probe_routes(monkeypatch)

    client = TestClient(app)
    response = client.get("/__gzip_probe/html", headers={"Accept-Encoding": "gzip"})

    assert response.status_code == 200
    assert response.headers.get("content-type", "").startswith("text/html")
    assert response.headers.get("content-encoding") == "gzip"
    assert len(response.content) > 1024


def test_real_server_app_does_not_gzip_server_sent_events(monkeypatch):
    app = _application_with_compression_probe_routes(monkeypatch)

    client = TestClient(app)
    response = client.get("/__gzip_probe/events", headers={"Accept-Encoding": "gzip"})

    assert response.status_code == 200
    assert response.headers.get("content-type", "").startswith("text/event-stream")
    assert "content-encoding" not in response.headers
    assert response.text == "data: update-0\n\ndata: update-1\n\ndata: update-2\n\n"
