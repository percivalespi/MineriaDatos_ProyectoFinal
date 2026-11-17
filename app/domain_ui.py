"""Constantes ejecutivas, traducción de alertas y utilidades geodésicas compartidas.

Centraliza la lógica de dominio para la API FastAPI (`app/api.py`) y el frontend
web (`app/static/`) sin duplicar reglas ni etiquetas.
"""

from __future__ import annotations

import math
from typing import Any

HISTORICAL_AVG_ETA_MIN = 26.3
_EARTH_RADIUS_KM = 6371.0

PRESET_SCENARIOS: dict[str, dict[str, Any]] = {
    "Escenario Ágil: Mañana Fluida (Perfil 0)": {
        "Ciudad_codigo": "BANG",
        "City": "Urban",
        "dist_haversine_km": 2.8,
        "Tiempo_preparacion_min": 10.0,
        "Time_Orderd": "10:15",
        "Order_Date": "22-03-2022",
        "Weather_conditions": "Sunny",
        "Road_traffic_density": "Low",
        "Festival": "No",
        "Delivery_person_Age": 26,
        "Delivery_person_Ratings": 4.9,
        "Vehicle_condition": 2,
        "multiple_deliveries": 0,
        "Type_of_vehicle": "electric_scooter",
        "Type_of_order": "Drinks",
    },
    "Escenario Habitual: Noche Estándar (Perfil 1)": {
        "Ciudad_codigo": "HYD",
        "City": "Metropolitan",
        "dist_haversine_km": 11.2,
        "Tiempo_preparacion_min": 10.0,
        "Time_Orderd": "19:45",
        "Order_Date": "25-03-2022",
        "Weather_conditions": "Cloudy",
        "Road_traffic_density": "Medium",
        "Festival": "No",
        "Delivery_person_Age": 29,
        "Delivery_person_Ratings": 4.7,
        "Vehicle_condition": 1,
        "multiple_deliveries": 1,
        "Type_of_vehicle": "scooter",
        "Type_of_order": "Meal",
    },
    "Escenario Crítico: Hora Pico y Alta Fricción (Perfil 2)": {
        "Ciudad_codigo": "MUM",
        "City": "Metropolitan",
        "dist_haversine_km": 14.5,
        "Tiempo_preparacion_min": 15.0,
        "Time_Orderd": "20:30",
        "Order_Date": "26-03-2022",
        "Weather_conditions": "Fog",
        "Road_traffic_density": "Jam",
        "Festival": "No",
        "Delivery_person_Age": 34,
        "Delivery_person_Ratings": 4.1,
        "Vehicle_condition": 0,
        "multiple_deliveries": 2,
        "Type_of_vehicle": "motorcycle",
        "Type_of_order": "Snack",
    },
}

CITY_LABELS: dict[str, str] = {
    "AGR": "Agra (AGR)",
    "ALH": "Allahabad (ALH)",
    "AURG": "Aurangabad (AURG)",
    "BANG": "Bangalore (BANG)",
    "BHP": "Bhopal (BHP)",
    "CHEN": "Chennai (CHEN)",
    "COIMB": "Coimbatore (COIMB)",
    "DEH": "Dehradun (DEH)",
    "GOA": "Goa (GOA)",
    "HYD": "Hyderabad (HYD)",
    "INDO": "Indore (INDO)",
    "JAP": "Jaipur (JAP)",
    "KNP": "Kanpur (KNP)",
    "KOC": "Kochi (KOC)",
    "KOL": "Kolkata (KOL)",
    "LUDH": "Ludhiana (LUDH)",
    "MUM": "Mumbai (MUM)",
    "MYS": "Mysore (MYS)",
    "PUNE": "Pune (PUNE)",
    "RANCHI": "Ranchi (RANCHI)",
    "SUR": "Surat (SUR)",
    "VAD": "Vadodara (VAD)",
}

CITY_TYPE_OPTIONS = ["Metropolitan", "Urban", "Semi-Urban"]
CITY_TYPE_LABELS: dict[str, str] = {
    "Metropolitan": "Área Metropolitana (Metropolitan)",
    "Urban": "Zona Urbana (Urban)",
    "Semi-Urban": "Zona Semiurbana / Periferia (Semi-Urban)",
}

TRAFFIC_OPTIONS = ["Low", "Medium", "High", "Jam"]
TRAFFIC_LABELS: dict[str, str] = {
    "Low": "Tráfico Fluido / Bajo (Low)",
    "Medium": "Tráfico Moderado (Medium)",
    "High": "Tráfico Pesado / Alto (High)",
    "Jam": "Congestión Severa / Embotellamiento (Jam)",
}

WEATHER_OPTIONS = ["Sunny", "Cloudy", "Windy", "Fog", "Sandstorms", "Stormy"]
WEATHER_LABELS: dict[str, str] = {
    "Sunny": "Soleado / Despejado (Sunny)",
    "Cloudy": "Nublado (Cloudy)",
    "Windy": "Ventoso (Windy)",
    "Fog": "Niebla (Fog)",
    "Sandstorms": "Tormenta de Arena (Sandstorms)",
    "Stormy": "Tormenta / Lluvia Fuerte (Stormy)",
}

FESTIVAL_OPTIONS = ["No", "Yes"]
FESTIVAL_LABELS: dict[str, str] = {
    "No": "Día Regular (No)",
    "Yes": "Día Festivo / Alta Demanda (Yes)",
}

VEHICLE_TYPE_OPTIONS = ["motorcycle", "scooter", "electric_scooter"]
VEHICLE_TYPE_LABELS: dict[str, str] = {
    "motorcycle": "Motocicleta (motorcycle)",
    "scooter": "Motoneta / Scooter (scooter)",
    "electric_scooter": "Scooter Eléctrico (electric_scooter)",
}

ORDER_TYPE_OPTIONS = ["Meal", "Snack", "Drinks", "Buffet"]
ORDER_TYPE_LABELS: dict[str, str] = {
    "Meal": "Comida Completa (Meal)",
    "Snack": "Botana / Antojo Rápido (Snack)",
    "Drinks": "Bebidas (Drinks)",
    "Buffet": "Pedido Familiar / Buffet (Buffet)",
}

EXECUTIVE_PROFILE_LABELS: dict[int, tuple[str, str]] = {
    0: (
        "Entrega Ágil Matutina (Perfil 0)",
        "Rutas cortas y tráfico ligero; operación ideal para entregas rápidas y comida recién hecha.",
    ),
    1: (
        "Operación Estándar Nocturna (Perfil 1)",
        "Volumen principal en horario de comida y cena con equilibrio estable entre distancia y repartidores.",
    ),
    2: (
        "Escenario de Alta Fricción Logística (Perfil 2)",
        "Concentra tráfico congestionado, múltiples paradas simultáneas o unidades con menor rendimiento.",
    ),
}


def transformar_distancia_a_haversine(
    ciudad_codigo: str,
    distancia_estimada_km: float,
    city_centroids: dict[str, tuple[float, float]],
) -> tuple[float, float, float, float]:
    """Transforma una distancia estimada en Kilómetros (km) a coordenadas geodésicas
    (Latitud y Longitud) cuya separación geodésica equivale a dicha distancia.
    """
    lat1_deg, lon1_deg = city_centroids.get(ciudad_codigo, (19.07, 72.87))
    lat1_r = math.radians(float(lat1_deg))
    lon1_r = math.radians(float(lon1_deg))
    delta = max(0.1, float(distancia_estimada_km)) / _EARTH_RADIUS_KM
    bearing = math.radians(45.0)

    sin_lat2 = math.sin(lat1_r) * math.cos(delta) + (
        math.cos(lat1_r) * math.sin(delta) * math.cos(bearing)
    )
    lat2_r = math.asin(max(-1.0, min(1.0, sin_lat2)))
    lon2_r = lon1_r + math.atan2(
        math.sin(bearing) * math.sin(delta) * math.cos(lat1_r),
        math.cos(delta) - math.sin(lat1_r) * math.sin(lat2_r),
    )
    return (
        float(lat1_deg),
        float(lon1_deg),
        round(math.degrees(lat2_r), 6),
        round(math.degrees(lon2_r), 6),
    )


def build_default_inputs(
    preset_name: str, imputer_artifacts: dict[str, Any]
) -> dict[str, Any]:
    """Inicializa valores editables del formulario a partir de un escenario base."""
    base = dict(PRESET_SCENARIOS[preset_name])
    centroids = imputer_artifacts.get("city_centroids", {})
    lat1, lon1, lat2, lon2 = transformar_distancia_a_haversine(
        base["Ciudad_codigo"], float(base["dist_haversine_km"]), centroids
    )
    base.update(
        {
            "use_coords": False,
            "rest_lat": lat1,
            "rest_lon": lon1,
            "del_lat": lat2,
            "del_lon": lon2,
        }
    )
    return base


def build_payload_from_inputs(
    form_inputs: dict[str, Any], city_centroids: dict[str, tuple[float, float]]
) -> dict[str, Any]:
    """Construye la entrada del modelo transformando la distancia estimada detrás de escena."""
    if form_inputs.get("use_coords", False):
        lat1 = float(form_inputs["rest_lat"])
        lon1 = float(form_inputs["rest_lon"])
        lat2 = float(form_inputs["del_lat"])
        lon2 = float(form_inputs["del_lon"])
    else:
        lat1, lon1, lat2, lon2 = transformar_distancia_a_haversine(
            str(form_inputs["Ciudad_codigo"]),
            float(form_inputs["dist_haversine_km"]),
            city_centroids,
        )

    return {
        "Ciudad_codigo": form_inputs["Ciudad_codigo"],
        "City": form_inputs["City"],
        "Time_Orderd": form_inputs["Time_Orderd"],
        "Tiempo_preparacion_min": float(form_inputs["Tiempo_preparacion_min"]),
        "Order_Date": form_inputs.get("Order_Date", "25-03-2022"),
        "Weather_conditions": form_inputs["Weather_conditions"],
        "Road_traffic_density": form_inputs["Road_traffic_density"],
        "Festival": form_inputs["Festival"],
        "Delivery_person_Age": int(form_inputs["Delivery_person_Age"]),
        "Delivery_person_Ratings": float(form_inputs["Delivery_person_Ratings"]),
        "Vehicle_condition": int(form_inputs["Vehicle_condition"]),
        "multiple_deliveries": int(form_inputs["multiple_deliveries"]),
        "Type_of_vehicle": form_inputs["Type_of_vehicle"],
        "Type_of_order": form_inputs["Type_of_order"],
        "Restaurant_latitude": lat1,
        "Restaurant_longitude": lon1,
        "Delivery_location_latitude": lat2,
        "Delivery_location_longitude": lon2,
    }


def format_executive_alert(raw_alert: str) -> tuple[str, bool]:
    """Traduce alertas técnicas a lenguaje claro para gerencia con abreviaturas explicadas."""
    lower = raw_alert.lower()
    if "nominal" in lower:
        return (
            "Operación saludable: no hay saturación vial ni alertas en la flota. "
            "Se recomienda mantener el Margen de Seguridad estándar (+1.5 Minutos (min) "
            "de Buffer SLA) para cumplir puntualmente con el cliente.",
            False,
        )
    if "+4.0 min" in lower:
        return (
            "Protección prioritaria de promesa al cliente (+4.0 Minutos (min) de Buffer SLA): "
            "el pedido coincide con tráfico congestionado, clima adverso o día festivo en un perfil "
            "exigente; ampliar la ventana mostrada al cliente previene reclamos y cancelaciones.",
            True,
        )
    if "+3.0 min" in lower:
        return (
            "Ajuste preventivo de tiempo prometido (+3.0 Minutos (min) de Buffer SLA): "
            "existe fricción operativa moderada en ruta o demanda alta; comunicar este margen "
            "adicional protege la puntualidad percibida.",
            True,
        )
    if "batching" in lower:
        return (
            "Evitar agrupar pedidos en un mismo viaje (Límite de Agrupamiento / Batching): "
            "asignar 2 o más entregas simultáneas en esta ruta congestionada o larga añade "
            "retrasos severos y enfría los alimentos; conviene despachar de forma individual.",
            True,
        )
    if "vehículo con condición 0" in lower or "mantenimiento" in lower:
        return (
            "Reasignar unidad de transporte: un vehículo en estado básico (condición 0) en "
            "una ruta mayor a 6 Kilómetros (km) eleva el riesgo de demora; se sugiere enviar "
            "una unidad en mejor estado mecánico (condición 1 o superior) y programar revisión.",
            True,
        )
    if "telemetría" in lower:
        return (
            "Sincronizar perfil de la persona repartidora: faltan datos vigentes en el sistema "
            "y se amplió el Margen de Seguridad (+4.5 Minutos (min) de Buffer SLA); "
            "verificar sesión activa antes del despacho mejora la precisión.",
            True,
        )
    return (raw_alert, True)


def build_executive_narrative(res: dict[str, Any]) -> str:
    """Genera una lectura narrativa breve comparando contra el promedio histórico."""
    eta = float(res["eta_predicho_min"])
    delta = eta - HISTORICAL_AVG_ETA_MIN
    cid = int(res["cluster_id"])
    if delta <= -2.0:
        comp = f"{abs(delta):.1f} Minutos (min) más rápido que el promedio histórico ({HISTORICAL_AVG_ETA_MIN:.1f} min)"
    elif delta >= 2.0:
        comp = f"{delta:.1f} Minutos (min) por encima del promedio histórico ({HISTORICAL_AVG_ETA_MIN:.1f} min)"
    else:
        comp = f"en línea con el promedio histórico de operación ({HISTORICAL_AVG_ETA_MIN:.1f} min)"

    if cid == 0:
        return (
            f"Entrega ágil ({comp}): alta probabilidad de llegada con temperatura óptima y máxima satisfacción del cliente."
        )
    if cid == 2:
        return (
            f"Atención prioritaria en despacho ({comp}): conviene proteger el tiempo prometido para evitar retrasos percibidos."
        )
    return (
        f"Desempeño estable ({comp}): ritmo controlado entre preparación en cocina y traslado final."
    )

