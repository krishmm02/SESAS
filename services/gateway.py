"""API Gateway: one front door for every service, plus the dashboard.

    uvicorn services.gateway:app --port 8000

Requests are routed on their first path segment (/roi/24510070100 -> ROIService on 8008)
and passed through unchanged, so each service stays independently callable on its own
port as well. GET responses are cached in process for CACHE_TTL seconds; X-Cache says
HIT or MISS. This stands in for Redis, which is first on the cut list - the scores only
change when a model is retrained, so a short TTL loses nothing.

The dashboard is served at /app/; every call it makes goes through this gateway.
"""

import os
import time

import requests
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from pipeline.config import ROOT

app = FastAPI(title="APIGateway")

# 127.0.0.1, not localhost: on Windows "localhost" tries IPv6 first and stalls ~2 s per call.
HOST = os.getenv("SERVICE_HOST", "http://127.0.0.1")
ROUTES = {
    "irradiance": 8001,
    "terrain": 8002,
    "socioecon": 8003,
    "adoption": 8004,
    "equity": 8005,
    "geojson": 8006,
    "geocode": 8007,  # GeocodingService serves /geocode, so that is its route prefix
    "roi": 8008,
    "developability": 8009,
    "yield": 8010,
    "lookup": 8006,  # VisualizationService: address -> geocoding -> equity
}
TIMEOUT = 30
CACHE_TTL = 300
_cache: dict[str, tuple[float, bytes, str]] = {}  # key -> (expires, body, content type)


@app.get("/health")
def health() -> dict:
    """Health of every routed service, so the failing one is named."""
    status = {}
    for name, port in ROUTES.items():
        try:
            status[name] = requests.get(f"{HOST}:{port}/health", timeout=5).json().get("status", "unknown")
        except requests.RequestException as e:
            status[name] = f"unreachable: {e.__class__.__name__}"
    return {"status": "ok", "services": status, "cache_entries": len(_cache)}


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse("/app/")


# Mounted before the catch-all route so /app/ is not proxied.
app.mount("/app", StaticFiles(directory=ROOT / "web", html=True), name="dashboard")


@app.get("/{path:path}")
def proxy(path: str, request: Request) -> Response:
    service = path.split("/", 1)[0]
    if service not in ROUTES:
        return JSONResponse({"detail": f"no service routes /{service}", "routes": list(ROUTES)}, 404)

    url = f"{HOST}:{ROUTES[service]}/{path}"
    key = f"{url}?{request.url.query}"
    hit = _cache.get(key)
    if hit and hit[0] > time.time():
        return Response(hit[1], media_type=hit[2], headers={"X-Cache": "HIT", "X-Upstream": service})

    try:
        r = requests.get(url, params=request.query_params, timeout=TIMEOUT)
    except requests.RequestException:
        return JSONResponse({"detail": f"{service} service unreachable on port {ROUTES[service]}"}, 503)
    ctype = r.headers.get("content-type", "application/json")
    if r.status_code == 200:
        _cache[key] = (time.time() + CACHE_TTL, r.content, ctype)
    return Response(r.content, status_code=r.status_code, media_type=ctype,
                    headers={"X-Cache": "MISS", "X-Upstream": service})
