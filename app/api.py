# API RESTful para la predicción de ETA y SLA de pedidos de delivery, con recomendaciones operativas para gerencia.

from __future__ import annotations

import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# Bibliotecas de FastAPI y Pydantic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.domain_ui import (  # noqa: E402
    CITY_LABELS,
    CITY_TYPE_LABELS,
    CITY_TYPE_OPTIONS,
    EXECUTIVE_PROFILE_LABELS,
    FESTIVAL_LABELS,
    FESTIVAL_OPTIONS,
    HISTORICAL_AVG_ETA_MIN,
    ORDER_TYPE_LABELS,
    ORDER_TYPE_OPTIONS,
    TRAFFIC_LABELS,
    TRAFFIC_OPTIONS,
    VEHICLE_TYPE_LABELS,
    VEHICLE_TYPE_OPTIONS,
    WEATHER_LABELS,
    WEATHER_OPTIONS,
    build_default_inputs,
    build_executive_narrative,
    build_payload_from_inputs,
    format_executive_alert,
)
from app.utils_inference import (  # noqa: E402
    get_available_cities,
    load_deployment_artifacts,
    predecir_pedido,
)

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_ARTIFACTS_CACHE: dict[str, Any] = {}


def _get_bundle() -> dict[str, Any]:
    """Carga y cachea el paquete de artefactos de inferencia en memoria."""
    if "bundle" not in _ARTIFACTS_CACHE:
        _ARTIFACTS_CACHE["bundle"] = load_deployment_artifacts()
    return _ARTIFACTS_CACHE["bundle"]


@asynccontextmanager
async def _lifespan(_: FastAPI):
    _get_bundle()
    yield


class OrderPredictionRequest(BaseModel):
    """Carga útil validada enviada desde el frontend ejecutivo."""

    ciudad_codigo: str = Field(default="HYD")
    city: str = Field(default="Metropolitan")
    use_coords: bool = Field(default=False)
    dist_haversine_km: float = Field(default=11.2, ge=0.5, le=30.0)
    rest_lat: float | None = Field(default=None)
    rest_lon: float | None = Field(default=None)
    del_lat: float | None = Field(default=None)
    del_lon: float | None = Field(default=None)
    time_orderd: str = Field(default="19:45")
    tiempo_preparacion_min: float = Field(default=10.0, ge=1.0, le=60.0)
    road_traffic_density: str = Field(default="Medium")
    weather_conditions: str = Field(default="Cloudy")
    festival: str = Field(default="No")
    delivery_person_age: int = Field(default=29, ge=16, le=70)
    delivery_person_ratings: float = Field(default=4.7, ge=1.0, le=5.0)
    vehicle_condition: int = Field(default=1, ge=0, le=3)
    multiple_deliveries: int = Field(default=1, ge=0, le=5)
    type_of_vehicle: str = Field(default="scooter")
    type_of_order: str = Field(default="Meal")

    @field_validator("time_orderd")
    @classmethod
    def _validate_hhmm(cls, value: str) -> str:
        clean = str(value).strip()
        if not re.fullmatch(r"(?:[01]?\d|2[0-3]):[0-5]\d", clean):
            raise ValueError(
                "La hora del pedido debe tener formato Horas y Minutos (HH:MM), por ejemplo 19:45."
            )
        hh, mm = clean.split(":")
        return f"{int(hh):02d}:{mm}"


app = FastAPI(
    title="API Ejecutiva de Estimación de Tiempo de Entrega (ETA)",
    version="1.0.0",
    lifespan=_lifespan,
)

# CORS Middleware para permitir solicitudes desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


def _to_options(keys: list[str], labels: dict[str, str]) -> list[dict[str, str]]:
    return [{"value": k, "label": labels.get(k, k)} for k in keys]


@app.get("/", include_in_schema=False)
def serve_frontend() -> FileResponse:
    """Entrega la interfaz web ejecutiva."""
    index_path = _STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend no encontrado.")
    return FileResponse(index_path)


@app.get("/api/health")
def health_check() -> dict[str, str]:
    """Verifica disponibilidad del servicio de predicción."""
    return {"status": "ok"}


@app.get("/api/metadata")
def get_metadata() -> dict[str, Any]:
    """Devuelve catálogos de opciones con abreviaturas explicadas y valores iniciales."""
    bundle = _get_bundle()
    imputer = bundle["imputer_artifacts"]
    cities = get_available_cities(imputer)
    centroids = {
        code: [round(float(coords[0]), 5), round(float(coords[1]), 5)]
        for code, coords in imputer.get("city_centroids", {}).items()
    }
    defaults = build_default_inputs(
        "Escenario Habitual: Noche Estándar (Perfil 1)", imputer
    )

    return {
        "historical_avg_eta_min": HISTORICAL_AVG_ETA_MIN,
        "cities": _to_options(cities, CITY_LABELS),
        "city_centroids": centroids,
        "city_types": _to_options(CITY_TYPE_OPTIONS, CITY_TYPE_LABELS),
        "traffic_levels": _to_options(TRAFFIC_OPTIONS, TRAFFIC_LABELS),
        "weather_conditions": _to_options(WEATHER_OPTIONS, WEATHER_LABELS),
        "festival_options": _to_options(FESTIVAL_OPTIONS, FESTIVAL_LABELS),
        "vehicle_types": _to_options(VEHICLE_TYPE_OPTIONS, VEHICLE_TYPE_LABELS),
        "order_types": _to_options(ORDER_TYPE_OPTIONS, ORDER_TYPE_LABELS),
        "defaults": {
            "ciudad_codigo": defaults["Ciudad_codigo"],
            "city": defaults["City"],
            "use_coords": bool(defaults["use_coords"]),
            "dist_haversine_km": float(defaults["dist_haversine_km"]),
            "rest_lat": float(defaults["rest_lat"]),
            "rest_lon": float(defaults["rest_lon"]),
            "del_lat": float(defaults["del_lat"]),
            "del_lon": float(defaults["del_lon"]),
            "time_orderd": defaults["Time_Orderd"],
            "tiempo_preparacion_min": float(defaults["Tiempo_preparacion_min"]),
            "road_traffic_density": defaults["Road_traffic_density"],
            "weather_conditions": defaults["Weather_conditions"],
            "festival": defaults["Festival"],
            "delivery_person_age": int(defaults["Delivery_person_Age"]),
            "delivery_person_ratings": float(defaults["Delivery_person_Ratings"]),
            "vehicle_condition": int(defaults["Vehicle_condition"]),
            "multiple_deliveries": int(defaults["multiple_deliveries"]),
            "type_of_vehicle": defaults["Type_of_vehicle"],
            "type_of_order": defaults["Type_of_order"],
        },
    }


@app.post("/api/predict")
def predict_delivery_eta(req: OrderPredictionRequest) -> dict[str, Any]:
    """Ejecuta la predicción de ETA, SLA y recomendaciones operativas para gerencia."""
    bundle = _get_bundle()
    imputer = bundle["imputer_artifacts"]
    centroids = imputer.get("city_centroids", {})

    form_inputs: dict[str, Any] = {
        "Ciudad_codigo": req.ciudad_codigo,
        "City": req.city,
        "use_coords": req.use_coords,
        "dist_haversine_km": req.dist_haversine_km,
        "rest_lat": req.rest_lat if req.rest_lat is not None else 17.385,
        "rest_lon": req.rest_lon if req.rest_lon is not None else 78.486,
        "del_lat": req.del_lat if req.del_lat is not None else 17.456,
        "del_lon": req.del_lon if req.del_lon is not None else 78.557,
        "Time_Orderd": req.time_orderd,
        "Tiempo_preparacion_min": req.tiempo_preparacion_min,
        "Order_Date": "25-03-2022",
        "Road_traffic_density": req.road_traffic_density,
        "Weather_conditions": req.weather_conditions,
        "Festival": req.festival,
        "Delivery_person_Age": req.delivery_person_age,
        "Delivery_person_Ratings": req.delivery_person_ratings,
        "Vehicle_condition": req.vehicle_condition,
        "multiple_deliveries": req.multiple_deliveries,
        "Type_of_vehicle": req.type_of_vehicle,
        "Type_of_order": req.type_of_order,
    }

    payload = build_payload_from_inputs(form_inputs, centroids)
    res = predecir_pedido(payload, artifacts_bundle=bundle)

    cid = int(res["cluster_id"])
    prof_title, prof_desc = EXECUTIVE_PROFILE_LABELS.get(
        cid, (f"Perfil Operativo {cid}", "Comportamiento operativo estándar.")
    )

    recomendaciones = []
    warn_count = 0
    for raw_alert in res["alertas_operativas"]:
        txt, is_warn = format_executive_alert(str(raw_alert))
        if is_warn:
            warn_count += 1
        recomendaciones.append({"texto": txt, "es_alerta": is_warn})

    buffer_val = float(res["buffer_sla_min"])
    if warn_count == 0:
        nivel_riesgo = "nominal"
    elif buffer_val >= 4.0 or warn_count >= 2:
        nivel_riesgo = "critico"
    else:
        nivel_riesgo = "moderado"

    eta_min = float(res["eta_predicho_min"])
    rango_min, rango_max = res["rango_estimado_min"]

    return {
        "eta_min": round(eta_min, 1),
        "sla_min": round(float(res["sla_recomendado_min"]), 1),
        "buffer_sla_min": round(buffer_val, 1),
        "rango_min": round(float(rango_min), 1),
        "rango_max": round(float(rango_max), 1),
        "perfil_id": cid,
        "perfil_titulo": prof_title,
        "perfil_descripcion": prof_desc,
        "dist_km": round(float(res["dist_haversine_km"]), 2),
        "tiempo_preparacion_min": round(float(res["tiempo_preparacion_min"]), 0),
        "promedio_historico_min": HISTORICAL_AVG_ETA_MIN,
        "diferencia_promedio_min": round(eta_min - HISTORICAL_AVG_ETA_MIN, 1),
        "nivel_riesgo": nivel_riesgo,
        "resumen_narrativo": build_executive_narrative(res),
        "recomendaciones": recomendaciones,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=False)

