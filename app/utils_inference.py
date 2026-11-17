"""Pipeline de inferencia para el despliegue en producción.

Se encadenan los 5 artefactos entrenados exclusivamente sobre `train`:
1. `models/imputer_artifacts.joblib`   (imputadores agrupados, modas y centroides por ciudad)
2. `models/scaler_clustering.joblib`   (`RobustScaler` sobre VARS_CLUSTER)
3. `models/clustering_model.joblib`    (`KMeans` K=3 operativo)
4. `models/preprocessor_reg.joblib`    (`ColumnTransformer` 41 variables COLS_NIVEL_3)
5. `models/best_eta_model.joblib`      (`ClusterSpecializedHybridRegressor` optimizado)
"""

from __future__ import annotations

import math
import pickle
import sys
from pathlib import Path
from typing import Any

# Importar scipy antes de scikit-learn para prevenir bloqueos transitorios de DLL en Windows.
import scipy.stats as stats  # noqa: F401
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODELS_DIR = _REPO_ROOT / "models"
_DEFAULT_DATA_DIR = _REPO_ROOT / "data"
_DEFAULT_REPORTS_DIR = _REPO_ROOT / "reports"

RANDOM_STATE = 42

_NULL_STRINGS = ["nan", "NaN", "None", "null", "<NA>", ""]
_TEXT_COLUMNS = [
    "ID",
    "Delivery_person_ID",
    "Order_Date",
    "Time_Orderd",
    "Time_Order_picked",
    "Weather_conditions",
    "Road_traffic_density",
    "Type_of_order",
    "Type_of_vehicle",
    "Festival",
    "City",
]

VARS_CLUSTER = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "dist_haversine_km",
    "Tiempo_preparacion_min",
    "Vehicle_condition",
    "Road_traffic_density",
    "multiple_deliveries",
    "hora_sin",
    "hora_cos",
]

CLUSTER_NAMES: dict[int, str] = {
    0: "Express urbano matutino",
    1: "Operación estándar nocturna",
    2: "Cuello de botella operativo",
}

COLS_NIVEL_2: list[str] = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "dist_haversine_km",
    "dist_manhattan_km",
    "Tiempo_preparacion_min",
    "dia_semana",
    "es_fin_semana",
    "hora_pedido",
    "hora_sin",
    "hora_cos",
    "Franja_Manana",
    "Franja_Almuerzo_Pico",
    "Franja_Tarde_Valle",
    "Franja_Cena_Pico",
    "Vehicle_condition",
    "multiple_deliveries",
    "Road_traffic_density",
    "Festival_Yes",
    "City_ord",
    "City_Semi-Urban",
    "City_Urban",
    "City_Unknown",
    "Weather_conditions",
    "Weather_Cloudy",
    "Weather_Fog",
    "Weather_Sandstorms",
    "Weather_Stormy",
    "Weather_Windy",
    "Type_of_order_Drinks",
    "Type_of_order_Meal",
    "Type_of_order_Snack",
    "Type_of_vehicle_motorcycle",
    "Type_of_vehicle_scooter",
    "Type_of_vehicle_electric_scooter",
]

COLS_NIVEL_3: list[str] = COLS_NIVEL_2 + [
    "cluster_label",
    "dist_centroide_1",
    "dist_centroide_2",
    "dist_centroide_3",
    "prob_pertenencia_1",
    "prob_pertenencia_2",
    "prob_pertenencia_3",
]


# ---------------------------------------------------------
# Componentes propios integrados en app/ (sin carpeta src/)
# ---------------------------------------------------------
def parse_time_to_minutes(val: Any) -> float:
    """Convierte HH:MM, HH:MM:SS, 24:XX:00 y decimales de Excel a minutos."""
    if pd.isna(val):
        return np.nan

    if isinstance(val, (int, float, np.number)):
        return float(int(round(float(val) * 1440)) % 1440)

    text = str(val).strip()
    if text in _NULL_STRINGS:
        return np.nan

    if ":" in text:
        parts = text.split(":")
        hours = int(parts[0]) % 24
        minutes = int(parts[1])
        return float(hours * 60 + minutes)

    try:
        return float(int(round(float(text) * 1440)) % 1440)
    except ValueError:
        return np.nan


def standardize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia espacios, normaliza nulos de texto y extrae Ciudad_codigo."""
    df = df.copy()
    for col in _TEXT_COLUMNS:
        if col in df.columns:
            cleaned = df[col].astype(str).str.strip()
            df[col] = cleaned.replace(_NULL_STRINGS, np.nan)

    if "Delivery_person_ID" in df.columns:
        extracted = (
            df["Delivery_person_ID"]
            .astype(str)
            .str.extract(r"^([A-Z]+)RES", expand=False)
        )
        if "Ciudad_codigo" in df.columns:
            df["Ciudad_codigo"] = df["Ciudad_codigo"].fillna(extracted).fillna("UNK")
        else:
            df["Ciudad_codigo"] = extracted.fillna("UNK")
    elif "Ciudad_codigo" not in df.columns:
        df["Ciudad_codigo"] = "UNK"

    if "City" in df.columns:
        df["City"] = df["City"].replace("Metropolitian", "Metropolitan")

    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte edades ilegales (<18) en NaN y trunca ratings a [1.0, 5.0]."""
    df = df.copy()
    if "Delivery_person_Age" in df.columns:
        df["Delivery_person_Age"] = pd.to_numeric(
            df["Delivery_person_Age"], errors="coerce"
        )
        df.loc[df["Delivery_person_Age"] < 18, "Delivery_person_Age"] = np.nan

    if "Delivery_person_Ratings" in df.columns:
        df["Delivery_person_Ratings"] = pd.to_numeric(
            df["Delivery_person_Ratings"], errors="coerce"
        )
        df["Delivery_person_Ratings"] = df["Delivery_person_Ratings"].clip(
            lower=1.0, upper=5.0
        )

    if "multiple_deliveries" in df.columns:
        df["multiple_deliveries"] = pd.to_numeric(
            df["multiple_deliveries"], errors="coerce"
        )

    return df


def fix_time_formats(df: pd.DataFrame) -> pd.DataFrame:
    """Crea columnas numéricas en minutos desde medianoche para pedido y recogida."""
    df = df.copy()
    if "Time_Orderd" in df.columns:
        df["Time_Orderd_min"] = df["Time_Orderd"].apply(parse_time_to_minutes)
    elif "Time_Orderd_min" not in df.columns:
        df["Time_Orderd_min"] = np.nan

    if "Time_Order_picked" in df.columns:
        df["Time_Order_picked_min"] = df["Time_Order_picked"].apply(
            parse_time_to_minutes
        )
    elif "Time_Order_picked_min" not in df.columns:
        df["Time_Order_picked_min"] = np.nan

    return df


def fix_coordinates(df: pd.DataFrame, artifacts: dict[str, Any]) -> pd.DataFrame:
    """Corrige signos invertidos mediante abs() y traslada (0,0) al centroide de Ciudad_codigo."""
    df = df.copy()
    coord_cols = [
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
    ]
    for col in coord_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").abs()

    mask_zero = (df["Restaurant_latitude"] < 1.0) | (df["Restaurant_longitude"] < 1.0)
    if mask_zero.any():
        centroids = artifacts["city_centroids"]
        global_lat, global_lon = artifacts["global_centroid"]

        city_lats = df.loc[mask_zero, "Ciudad_codigo"].map(
            lambda c: centroids.get(str(c), (global_lat, global_lon))[0]
        )
        city_lons = df.loc[mask_zero, "Ciudad_codigo"].map(
            lambda c: centroids.get(str(c), (global_lat, global_lon))[1]
        )

        df.loc[mask_zero, "Restaurant_latitude"] = (
            df.loc[mask_zero, "Restaurant_latitude"] + city_lats.values
        )
        df.loc[mask_zero, "Delivery_location_latitude"] = (
            df.loc[mask_zero, "Delivery_location_latitude"] + city_lats.values
        )
        df.loc[mask_zero, "Restaurant_longitude"] = (
            df.loc[mask_zero, "Restaurant_longitude"] + city_lons.values
        )
        df.loc[mask_zero, "Delivery_location_longitude"] = (
            df.loc[mask_zero, "Delivery_location_longitude"] + city_lons.values
        )

    return df


def handle_missing_values(df: pd.DataFrame, artifacts: dict[str, Any]) -> pd.DataFrame:
    """Imputa nulos numéricos, categóricos y temporales usando artefactos de entrenamiento."""
    df = df.copy()

    if "Delivery_person_Age" in df.columns:
        age_fill = (
            df["Ciudad_codigo"]
            .map(artifacts["age_median_by_city"])
            .fillna(artifacts["age_median_global"])
        )
        df["Delivery_person_Age"] = df["Delivery_person_Age"].fillna(age_fill)

    if "Delivery_person_Ratings" in df.columns:
        ratings_fill = (
            df["Ciudad_codigo"]
            .map(artifacts["ratings_median_by_city"])
            .fillna(artifacts["ratings_median_global"])
        )
        df["Delivery_person_Ratings"] = df["Delivery_person_Ratings"].fillna(
            ratings_fill
        )

    if "multiple_deliveries" in df.columns:
        df["multiple_deliveries"] = df["multiple_deliveries"].fillna(
            artifacts["multiple_deliveries_mode"]
        )

    if "Festival" in df.columns:
        df["Festival"] = df["Festival"].fillna("No")

    if "City" in df.columns:
        df["City"] = df["City"].fillna("Unknown")

    if "Road_traffic_density" in df.columns:
        df["Road_traffic_density"] = df["Road_traffic_density"].fillna(
            artifacts["traffic_mode"]
        )

    if "Weather_conditions" in df.columns:
        df["Weather_conditions"] = df["Weather_conditions"].fillna(
            artifacts["weather_mode"]
        )

    prep_median = artifacts.get("prep_time_median", 10.0)
    mask_orderd_na = df["Time_Orderd_min"].isna()
    if mask_orderd_na.any():
        picked_vals = df.loc[mask_orderd_na, "Time_Order_picked_min"].fillna(730.0)
        df.loc[mask_orderd_na, "Time_Orderd_min"] = (
            picked_vals - prep_median
        ) % 1440

    mask_picked_na = df["Time_Order_picked_min"].isna()
    if mask_picked_na.any():
        df.loc[mask_picked_na, "Time_Order_picked_min"] = (
            df.loc[mask_picked_na, "Time_Orderd_min"] + prep_median
        ) % 1440

    df = df.drop(columns=["Time_Orderd", "Time_Order_picked"], errors="ignore")
    return df


def clean_data(
    df: pd.DataFrame,
    artifacts: dict[str, Any] | None = None,
    fit: bool = False,
    artifacts_path: str | Path | None = None,
) -> pd.DataFrame:
    """Pipeline de limpieza para inferencia reutilizando los artefactos de entrenamiento."""
    if fit:
        raise ValueError("La app de inferencia solo opera con fit=False.")
    if artifacts is None:
        resolved = (
            _DEFAULT_MODELS_DIR / "imputer_artifacts.joblib"
            if artifacts_path is None
            else Path(artifacts_path)
        )
        with open(resolved, "rb") as handle:
            artifacts = pickle.load(handle)

    df = df.copy()
    df = standardize_categoricals(df)
    df = handle_outliers(df)
    df = fix_time_formats(df)
    df = fix_coordinates(df, artifacts)
    df = handle_missing_values(df, artifacts)
    return df


def haversine_distance(lat1: Any, lon1: Any, lat2: Any, lon2: Any) -> Any:
    """Calcula la distancia geodésica en km usando la fórmula de Haversine."""
    earth_radius_km = 6371.0
    lat1_r, lon1_r, lat2_r, lon2_r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arcsin(np.sqrt(a))
    return earth_radius_km * c


def calculate_distances(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula dist_haversine_km y dist_manhattan_km ponderada por cos(lat_media)."""
    df = df.copy()
    lat1 = df["Restaurant_latitude"].astype(float)
    lon1 = df["Restaurant_longitude"].astype(float)
    lat2 = df["Delivery_location_latitude"].astype(float)
    lon2 = df["Delivery_location_longitude"].astype(float)

    df["dist_haversine_km"] = haversine_distance(lat1, lon1, lat2, lon2)

    mean_lat_rad = np.radians((lat1 + lat2) / 2.0)
    delta_lat_km = np.abs(lat2 - lat1) * 111.132
    delta_lon_km = np.abs(lon2 - lon1) * 111.320 * np.cos(mean_lat_rad)
    df["dist_manhattan_km"] = delta_lat_km + delta_lon_km
    return df


def calculate_prep_time(
    df: pd.DataFrame, prep_time_median: float = 10.0
) -> pd.DataFrame:
    """Calcula Tiempo_preparacion_min corrigiendo cruces de medianoche y valores >180 min."""
    df = df.copy()
    prep = df["Time_Order_picked_min"] - df["Time_Orderd_min"]
    prep = np.where(prep < 0, prep + 1440.0, prep)
    prep = pd.Series(prep, index=df.index, dtype=float)
    prep = prep.where(prep <= 180.0, np.nan).fillna(float(prep_time_median))
    df["Tiempo_preparacion_min"] = prep
    return df


def _assign_franja_horaria(hour: int) -> str:
    if 0 <= hour < 6:
        return "Madrugada"
    if 6 <= hour < 12:
        return "Manana"
    if 12 <= hour < 16:
        return "Almuerzo_Pico"
    if 16 <= hour < 19:
        return "Tarde_Valle"
    return "Cena_Pico"


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae día de semana, fin de semana, hora, seno/coseno cíclicos y franja horaria."""
    df = df.copy()
    order_dt = pd.to_datetime(df["Order_Date"], format="%d-%m-%Y", errors="coerce")

    df["dia_semana"] = order_dt.dt.dayofweek.fillna(3).astype(int)
    df["es_fin_semana"] = (df["dia_semana"] >= 5).astype(int)

    df["hora_pedido"] = ((df["Time_Orderd_min"] // 60) % 24).astype(float)
    df["hora_sin"] = np.sin(2.0 * np.pi * df["hora_pedido"] / 24.0)
    df["hora_cos"] = np.cos(2.0 * np.pi * df["hora_pedido"] / 24.0)
    df["Franja_horaria"] = df["hora_pedido"].astype(int).map(_assign_franja_horaria)

    return df.drop(
        columns=["Order_Date", "Time_Orderd_min", "Time_Order_picked_min"],
        errors="ignore",
    )


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica codificación ordinal lógica y One-Hot determinista."""
    df = df.copy()

    traffic_map = {"Low": 1, "Medium": 2, "High": 3, "Jam": 4}
    city_ord_map = {"Semi-Urban": 1, "Urban": 2, "Metropolitan": 3, "Unknown": 2}
    weather_map = {
        "Sunny": 0,
        "Cloudy": 1,
        "Windy": 2,
        "Fog": 3,
        "Sandstorms": 4,
        "Stormy": 5,
    }

    if "Weather_conditions" in df.columns:
        weather_vals = df["Weather_conditions"].astype(str).str.strip()
        for cat in ["Cloudy", "Fog", "Sandstorms", "Stormy", "Windy"]:
            df[f"Weather_{cat}"] = (weather_vals == cat).astype(int)
        df["Weather_conditions"] = weather_vals.map(weather_map).fillna(0).astype(int)

    if "Road_traffic_density" in df.columns:
        traffic_vals = df["Road_traffic_density"].astype(str).str.strip()
        df["Road_traffic_density"] = traffic_vals.map(traffic_map).fillna(2).astype(int)

    if "City" in df.columns:
        city_vals = (
            df["City"].astype(str).str.strip().replace("Metropolitian", "Metropolitan")
        )
        df["City_ord"] = city_vals.map(city_ord_map).fillna(2).astype(int)
        for cat in ["Semi-Urban", "Urban", "Unknown"]:
            df[f"City_{cat}"] = (city_vals == cat).astype(int)

    if "Type_of_order" in df.columns:
        order_vals = df["Type_of_order"].astype(str).str.strip()
        for cat in ["Drinks", "Meal", "Snack"]:
            df[f"Type_of_order_{cat}"] = (order_vals == cat).astype(int)

    if "Type_of_vehicle" in df.columns:
        vehicle_vals = df["Type_of_vehicle"].astype(str).str.strip()
        for cat in ["motorcycle", "scooter", "electric_scooter"]:
            df[f"Type_of_vehicle_{cat}"] = (vehicle_vals == cat).astype(int)

    if "Festival" in df.columns:
        festival_vals = df["Festival"].astype(str).str.strip()
        df["Festival_Yes"] = (festival_vals == "Yes").astype(int)

    if "Franja_horaria" in df.columns:
        franja_vals = df["Franja_horaria"].astype(str).str.strip()
        for cat in ["Manana", "Almuerzo_Pico", "Tarde_Valle", "Cena_Pico"]:
            df[f"Franja_{cat}"] = (franja_vals == cat).astype(int)

    return df.drop(
        columns=["Type_of_order", "Type_of_vehicle", "Festival", "City", "Franja_horaria"],
        errors="ignore",
    )


def engineer_features(df: pd.DataFrame, prep_time_median: float = 10.0) -> pd.DataFrame:
    """Pipeline determinista de ingeniería de características para inferencia."""
    df = df.copy()
    df = calculate_distances(df)
    df = calculate_prep_time(df, prep_time_median=prep_time_median)
    df = extract_temporal_features(df)
    df = encode_categoricals(df)
    return df


def add_cluster_features(
    df: pd.DataFrame, X_scaled: Any, kmeans: Any, scaler: Any = None
) -> pd.DataFrame:
    """Añade etiqueta de cluster, distancias euclidianas a centroides y probabilidades."""
    df = df.copy()
    arr = np.asarray(X_scaled, dtype=float)
    k = kmeans.n_clusters

    df["cluster_label"] = kmeans.predict(arr)
    distances = kmeans.transform(arr)
    for i in range(k):
        df[f"dist_centroide_{i + 1}"] = distances[:, i]

    eps = 1e-6
    inv_dist = 1.0 / (distances + eps)
    probs = inv_dist / inv_dist.sum(axis=1, keepdims=True)
    for i in range(k):
        df[f"prob_pertenencia_{i + 1}"] = probs[:, i]

    return df


class ClusterSpecializedHybridRegressor(BaseEstimator, RegressorMixin):
    """Regresor híbrido: combina modelo global y especialistas por cluster."""

    def __init__(
        self,
        global_params: dict[str, Any] | None = None,
        cluster_params: dict[str, Any] | None = None,
        global_weight: float = 0.70,
        cluster_col: str = "cluster_label",
        global_cols: list[str] | None = None,
        cluster_cols: list[str] | None = None,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.global_params = global_params
        self.cluster_params = cluster_params
        self.global_weight = float(global_weight)
        self.cluster_col = cluster_col
        self.global_cols = global_cols
        self.cluster_cols = cluster_cols
        self.random_state = int(random_state)

    def _resolve_params(self) -> tuple[dict[str, Any], dict[str, Any]]:
        default_global = {
            "max_iter": 300,
            "max_depth": 10,
            "max_leaf_nodes": 25,
            "min_samples_leaf": 60,
            "learning_rate": 0.05,
            "l2_regularization": 2.0,
            "random_state": self.random_state,
        }
        default_cluster = {
            "max_iter": 250,
            "max_depth": 8,
            "max_leaf_nodes": 20,
            "min_samples_leaf": 40,
            "learning_rate": 0.04,
            "l2_regularization": 5.0,
            "random_state": self.random_state,
        }
        g_params = {**default_global, **(self.global_params or {})}
        c_params = {**default_cluster, **(self.cluster_params or {})}
        return g_params, c_params

    def _prepare_indices(self, X: pd.DataFrame | np.ndarray) -> None:
        if isinstance(X, pd.DataFrame):
            cols = list(X.columns)
            self.feature_names_in_ = np.asarray(cols, dtype=object)
            self.n_features_in_ = len(cols)
        else:
            cols = list(self.cluster_cols or COLS_NIVEL_3)
            self.feature_names_in_ = np.asarray(cols, dtype=object)
            self.n_features_in_ = X.shape[1]

        g_target = list(self.global_cols or COLS_NIVEL_2)
        c_target = [
            c for c in (self.cluster_cols or COLS_NIVEL_3) if c != self.cluster_col
        ]

        self.global_indices_ = [cols.index(c) for c in g_target if c in cols]
        self.cluster_indices_ = [cols.index(c) for c in c_target if c in cols]
        if not self.global_indices_:
            self.global_indices_ = list(range(self.n_features_in_))
        if not self.cluster_indices_:
            self.cluster_indices_ = list(range(self.n_features_in_))
        if self.cluster_col not in cols:
            raise ValueError(
                f"La columna '{self.cluster_col}' es requerida por ClusterSpecializedHybridRegressor."
            )
        self.cluster_col_idx_ = cols.index(self.cluster_col)

    def fit(
        self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray
    ) -> "ClusterSpecializedHybridRegressor":
        self._prepare_indices(X)
        arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        g_params, c_params = self._resolve_params()
        self.global_model_ = HistGradientBoostingRegressor(**g_params)
        self.global_model_.fit(arr[:, self.global_indices_], y_arr)

        clusters = np.round(arr[:, self.cluster_col_idx_]).astype(int)
        self.clusters_ = np.sort(np.unique(clusters))
        self.cluster_models_: dict[int, HistGradientBoostingRegressor] = {}

        for cluster_id in self.clusters_:
            mask = clusters == cluster_id
            model_k = HistGradientBoostingRegressor(**c_params)
            model_k.fit(arr[mask][:, self.cluster_indices_], y_arr[mask])
            self.cluster_models_[int(cluster_id)] = model_k

        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        pred_global = self.global_model_.predict(arr[:, self.global_indices_])

        clusters = np.round(arr[:, self.cluster_col_idx_]).astype(int)
        pred_cluster = pred_global.copy()

        for cluster_id, model_k in self.cluster_models_.items():
            mask = clusters == cluster_id
            if np.any(mask):
                pred_cluster[mask] = model_k.predict(
                    arr[mask][:, self.cluster_indices_]
                )

        w = self.global_weight
        return w * pred_global + (1.0 - w) * pred_cluster


def _register_hybrid_class_for_unpickling() -> None:
    """Expone `ClusterSpecializedHybridRegressor` en `__main__` para cargar joblib sin `src`."""
    main_mod = sys.modules.get("__main__")
    if main_mod is not None and not hasattr(
        main_mod, "ClusterSpecializedHybridRegressor"
    ):
        setattr(
            main_mod,
            "ClusterSpecializedHybridRegressor",
            ClusterSpecializedHybridRegressor,
        )


_register_hybrid_class_for_unpickling()

_REQUIRED_ARTIFACTS = {
    "imputer_artifacts": "imputer_artifacts.joblib",
    "scaler_clustering": "scaler_clustering.joblib",
    "clustering_model": "clustering_model.joblib",
    "preprocessor_reg": "preprocessor_reg.joblib",
    "best_eta_model": "best_eta_model.joblib",
}

_RAW_INPUT_DEFAULTS: dict[str, Any] = {
    "ID": "SIM_0001",
    "Delivery_person_ID": "BANG_RES01DEL01",
    "Delivery_person_Age": 29.0,
    "Delivery_person_Ratings": 4.7,
    "Restaurant_latitude": np.nan,
    "Restaurant_longitude": np.nan,
    "Delivery_location_latitude": np.nan,
    "Delivery_location_longitude": np.nan,
    "Order_Date": "25-03-2022",
    "Time_Orderd": "19:30",
    "Time_Order_picked": np.nan,
    "Weather_conditions": "Sunny",
    "Road_traffic_density": "Medium",
    "Vehicle_condition": 1,
    "Type_of_order": "Meal",
    "Type_of_vehicle": "motorcycle",
    "multiple_deliveries": 1.0,
    "Festival": "No",
    "City": "Metropolitan",
}


def _load_single_artifact(path: Path) -> Any:
    _register_hybrid_class_for_unpickling()
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as handle:
            return pickle.load(handle)


def load_deployment_artifacts(models_dir: str | Path | None = None) -> dict[str, Any]:
    """Carga los 5 artefactos de `models/` requeridos para el despliegue end-to-end."""
    base_dir = _DEFAULT_MODELS_DIR if models_dir is None else Path(models_dir)
    bundle: dict[str, Any] = {}
    missing = []

    for key, filename in _REQUIRED_ARTIFACTS.items():
        artifact_path = base_dir / filename
        if not artifact_path.exists():
            missing.append(str(artifact_path))
        else:
            bundle[key] = _load_single_artifact(artifact_path)

    if missing:
        raise FileNotFoundError(
            f"Faltan artefactos requeridos en disco para inferencia: {missing}"
        )

    bundle["models_dir"] = str(base_dir)
    return bundle


def _format_minutes_hhmm(minutes_from_midnight: float) -> str:
    mins = int(round(minutes_from_midnight)) % 1440
    return f"{mins // 60:02d}:{mins % 60:02d}"


def _synthesize_coords_from_distance(
    city_code: str, dist_km: float, imputer_artifacts: dict[str, Any]
) -> tuple[float, float, float, float]:
    """Genera coordenadas geodésicas consistentes desde el centroide de ciudad cuando solo
    se provee `dist_haversine_km` en modo simulación."""
    centroids = imputer_artifacts.get("city_centroids", {})
    global_centroid = imputer_artifacts.get("global_centroid", (19.07, 72.87))
    lat1, lon1 = centroids.get(str(city_code), global_centroid)
    dist = max(float(dist_km), 0.1)
    # Desplazamiento diagonal de 45 grados aproximando relación Manhattan/Haversine ~ sqrt(2)
    step_km = dist / math.sqrt(2.0)
    dlat = step_km / 111.132
    dlon = step_km / (111.320 * max(math.cos(math.radians(lat1)), 0.1))
    return lat1, lon1, lat1 + dlat, lon1 + dlon


def _prepare_raw_input_df(
    datos_nuevos: dict[str, Any] | list[dict[str, Any]] | pd.DataFrame,
    imputer_artifacts: dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Normaliza la entrada cruda o de simulación y audita faltantes de telemetría en origen."""
    if isinstance(datos_nuevos, dict):
        df = pd.DataFrame([datos_nuevos])
    elif isinstance(datos_nuevos, list):
        df = pd.DataFrame(datos_nuevos)
    elif isinstance(datos_nuevos, pd.DataFrame):
        df = datos_nuevos.copy()
    else:
        raise TypeError("`datos_nuevos` debe ser dict, list[dict] o pd.DataFrame.")

    if len(df) == 0:
        raise ValueError("La entrada `datos_nuevos` no contiene registros.")

    # Auditar telemetría faltante ANTES de imputar (hallazgo crítico de Fase 5)
    ratings_series = df.get("Delivery_person_Ratings", pd.Series(np.nan, index=df.index))
    age_series = df.get("Delivery_person_Age", pd.Series(np.nan, index=df.index))
    orderd_series = df.get("Time_Orderd", pd.Series(np.nan, index=df.index))
    telemetry_missing = (
        pd.to_numeric(ratings_series, errors="coerce").isna()
        | pd.to_numeric(age_series, errors="coerce").isna()
        | orderd_series.isna()
    ).to_numpy()

    # Si es entrada manual desde calculadora (faltan columnas crudas), completar con valores seguros
    for col, default_val in _RAW_INPUT_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default_val

    for t_col in ("Time_Orderd", "Time_Order_picked"):
        if t_col in df.columns:
            df[t_col] = df[t_col].astype(object)
    for c_col in (
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
    ):
        if c_col in df.columns:
            df[c_col] = pd.to_numeric(df[c_col], errors="coerce").astype(float)

    # Si el usuario especificó Ciudad_codigo explícitamente sin Delivery_person_ID válido
    if "Ciudad_codigo" in df.columns:
        df["Ciudad_codigo"] = df["Ciudad_codigo"].astype(str).str.strip().str.upper()
        mask_synthetic_id = df["Delivery_person_ID"].astype(str) == "BANG_RES01DEL01"
        if mask_synthetic_id.any():
            df.loc[mask_synthetic_id, "Delivery_person_ID"] = (
                df.loc[mask_synthetic_id, "Ciudad_codigo"] + "RES01DEL01"
            )

    # Modo simulación rápida: construir Time_Order_picked a partir de Tiempo_preparacion_min si falta
    if "Tiempo_preparacion_min" in df.columns:
        prep_series = pd.to_numeric(df["Tiempo_preparacion_min"], errors="coerce")
        needs_picked = df["Time_Order_picked"].isna() & prep_series.notna()
        for idx in df.index[needs_picked]:
            base_min = parse_time_to_minutes(df.at[idx, "Time_Orderd"])
            if pd.isna(base_min):
                base_min = 1170.0  # 19:30 por defecto
                df.at[idx, "Time_Orderd"] = "19:30"
            df.at[idx, "Time_Order_picked"] = _format_minutes_hhmm(
                base_min + float(prep_series.at[idx])
            )

    # Modo simulación rápida: construir coordenadas desde dist_haversine_km si faltan lat/lon
    if "dist_haversine_km" in df.columns:
        dist_series = pd.to_numeric(df["dist_haversine_km"], errors="coerce")
        coord_missing = (
            pd.to_numeric(df["Restaurant_latitude"], errors="coerce").isna()
            | pd.to_numeric(df["Delivery_location_latitude"], errors="coerce").isna()
        )
        needs_coords = coord_missing & dist_series.notna()
        for idx in df.index[needs_coords]:
            if "Ciudad_codigo" in df.columns:
                city_val = str(df.at[idx, "Ciudad_codigo"])
            else:
                raw_id = str(df.at[idx, "Delivery_person_ID"])
                city_val = raw_id.split("RES")[0] if "RES" in raw_id else "BANG"
            lat1, lon1, lat2, lon2 = _synthesize_coords_from_distance(
                city_val, float(dist_series.at[idx]), imputer_artifacts
            )
            df.at[idx, "Restaurant_latitude"] = lat1
            df.at[idx, "Restaurant_longitude"] = lon1
            df.at[idx, "Delivery_location_latitude"] = lat2
            df.at[idx, "Delivery_location_longitude"] = lon2

    # Respaldo de coordenadas globales si aún hubiera NaN tras simulación
    for col, fallback_idx in [
        ("Restaurant_latitude", 0),
        ("Restaurant_longitude", 1),
        ("Delivery_location_latitude", 0),
        ("Delivery_location_longitude", 1),
    ]:
        if pd.to_numeric(df[col], errors="coerce").isna().any():
            global_lat, global_lon = imputer_artifacts["global_centroid"]
            fill_c = global_lat if fallback_idx == 0 else global_lon
            offset = 0.04 if "Delivery_location" in col else 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(fill_c + offset)

    return df, telemetry_missing


def _compute_hybrid_components(
    model: ClusterSpecializedHybridRegressor, X_cols3: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula la predicción híbrida junto con los componentes global y especialista."""
    arr = np.asarray(X_cols3, dtype=float)
    pred_global = model.global_model_.predict(arr[:, model.global_indices_])

    clusters = np.round(arr[:, model.cluster_col_idx_]).astype(int)
    pred_cluster = pred_global.copy()
    for cid, submodel in model.cluster_models_.items():
        mask = clusters == cid
        if np.any(mask):
            pred_cluster[mask] = submodel.predict(arr[mask][:, model.cluster_indices_])

    w = float(model.global_weight)
    pred_final = w * pred_global + (1.0 - w) * pred_cluster
    return pred_final, pred_global, pred_cluster


def _build_operational_guidance(
    row: pd.Series,
    eta_pred: float,
    telemetry_missing: bool,
) -> tuple[float, list[str]]:
    """Aplica las reglas operativas de negocio derivadas de la Fase 5 (SLA + alertas)."""
    cluster_id = int(row["cluster_label"])
    traffic_ord = int(row["Road_traffic_density"])
    multi = float(row["multiple_deliveries"])
    veh_cond = int(row["Vehicle_condition"])
    festival = int(row["Festival_Yes"])
    dist_km = float(row["dist_haversine_km"])
    bad_weather = bool(row.get("Weather_Fog", 0) or row.get("Weather_Cloudy", 0) or row.get("Weather_Stormy", 0))

    buffer_min = 1.5
    alerts: list[str] = []

    if cluster_id == 2 and (bad_weather or festival == 1 or traffic_ord >= 4):
        buffer_min = 4.0
        alerts.append(
            "Buffer SLA ampliado (+4.0 min): pedido clasificado en Cluster 2 (Cuello de botella) con clima/tráfico o festivo crítico."
        )
    elif cluster_id == 2 or traffic_ord >= 4 or festival == 1:
        buffer_min = 3.0
        alerts.append(
            "Buffer SLA preventivo (+3.0 min): escenario de alta fricción operativa (Cluster 2, tráfico Jam o evento Festival)."
        )

    if multi >= 2 and (cluster_id == 2 or traffic_ord >= 4 or dist_km > 12.0):
        alerts.append(
            "Regla de despacho (Límite de Batching): evitar agrupar >= 2 pedidos en esta ruta para prevenir recargos extremos (+4 a +9 min)."
        )

    if veh_cond == 0 and dist_km > 6.0:
        alerts.append(
            "Asignación de flota (Mantenimiento): vehículo con condición 0 asignado a ruta > 6 km; priorizar unidad con condición >= 1."
        )

    if telemetry_missing:
        buffer_min = max(buffer_min, 4.5)
        alerts.append(
            "Alerta de telemetría en origen: faltan edad/rating/hora real del repartidor; exigir sincronización en app antes del despacho."
        )

    if not alerts:
        alerts.append(
            "Operación nominal: sin alertas de congestión severa ni restricciones de flota; mantener SLA estándar (+1.5 min)."
        )

    return round(eta_pred + buffer_min, 2), alerts


def predecir_pedido(
    datos_nuevos: dict[str, Any] | list[dict[str, Any]] | pd.DataFrame,
    artifacts_bundle: dict[str, Any] | None = None,
    return_details: bool = True,
) -> dict[str, Any] | pd.DataFrame:
    """Pipeline unificado de inferencia sin fuga de datos:
    datos crudos -> limpieza (`clean_data`) -> FE (`engineer_features`) ->
    escalado y clustering (`add_cluster_features`) -> preprocesamiento y regresión híbrida.
    """
    bundle = load_deployment_artifacts() if artifacts_bundle is None else artifacts_bundle
    is_single_dict = isinstance(datos_nuevos, dict)

    df_raw, telemetry_missing = _prepare_raw_input_df(
        datos_nuevos, bundle["imputer_artifacts"]
    )

    # 1. Limpieza e imputación con artefactos de train (fit=False)
    df_clean = clean_data(df_raw, artifacts=bundle["imputer_artifacts"], fit=False)

    # 2. Ingeniería de características determinista
    prep_median = float(bundle["imputer_artifacts"].get("prep_time_median", 10.0))
    df_fe = engineer_features(df_clean, prep_time_median=prep_median)

    # 3. Escalado y asignación de perfil operativo (Clustering)
    X_cluster_scaled = bundle["scaler_clustering"].transform(
        df_fe[VARS_CLUSTER].to_numpy(dtype=float)
    )
    df_clustered = add_cluster_features(
        df_fe, X_cluster_scaled, bundle["clustering_model"]
    )

    # 4. Validación del ColumnTransformer y predicción híbrida final
    X_reg_df = df_clustered[COLS_NIVEL_3]
    _ = bundle["preprocessor_reg"].transform(X_reg_df)

    eta_pred, eta_global, eta_cluster = _compute_hybrid_components(
        bundle["best_eta_model"], X_reg_df
    )

    df_clustered["eta_predicho_min"] = eta_pred
    df_clustered["eta_global_min"] = eta_global
    df_clustered["eta_especialista_min"] = eta_cluster
    df_clustered["cluster_nombre"] = df_clustered["cluster_label"].map(CLUSTER_NAMES)

    if not is_single_dict:
        if not return_details:
            return df_clustered[
                [
                    "cluster_label",
                    "cluster_nombre",
                    "eta_predicho_min",
                    "eta_global_min",
                    "eta_especialista_min",
                ]
            ]
        return df_clustered

    # Salida detallada estructurada para un pedido individual (dict)
    row = df_clustered.iloc[0]
    cid = int(row["cluster_label"])
    eta_val = float(row["eta_predicho_min"])
    sla_rec, alerts = _build_operational_guidance(
        row, eta_val, bool(telemetry_missing[0])
    )
    buffer_val = round(sla_rec - eta_val, 2)

    return {
        "eta_predicho_min": round(eta_val, 2),
        "eta_global_min": round(float(row["eta_global_min"]), 2),
        "eta_especialista_min": round(float(row["eta_especialista_min"]), 2),
        "global_weight": float(bundle["best_eta_model"].global_weight),
        "sla_recomendado_min": sla_rec,
        "buffer_sla_min": buffer_val,
        "rango_estimado_min": (
            round(max(8.0, eta_val - 3.1), 1),
            round(eta_val + max(buffer_val, 3.1), 1),
        ),
        "cluster_id": cid,
        "cluster_nombre": CLUSTER_NAMES.get(cid, f"Cluster {cid}"),
        "distancias_centroides": {
            "Cluster 0": round(float(row["dist_centroide_1"]), 3),
            "Cluster 1": round(float(row["dist_centroide_2"]), 3),
            "Cluster 2": round(float(row["dist_centroide_3"]), 3),
        },
        "probabilidades_pertenencia": {
            "Cluster 0": round(float(row["prob_pertenencia_1"]), 4),
            "Cluster 1": round(float(row["prob_pertenencia_2"]), 4),
            "Cluster 2": round(float(row["prob_pertenencia_3"]), 4),
        },
        "dist_haversine_km": round(float(row["dist_haversine_km"]), 2),
        "dist_manhattan_km": round(float(row["dist_manhattan_km"]), 2),
        "tiempo_preparacion_min": round(float(row["Tiempo_preparacion_min"]), 1),
        "telemetry_missing": bool(telemetry_missing[0]),
        "alertas_operativas": alerts,
        "features_procesadas": df_clustered[COLS_NIVEL_3].iloc[0].to_dict(),
    }


def get_fleet_dashboard_data() -> dict[str, pd.DataFrame]:
    """Carga y consolida tablas agregadas para el Dashboard Ejecutivo de Flota."""
    eval_cluster = pd.read_csv(_DEFAULT_REPORTS_DIR / "evaluacion_por_cluster.csv")
    eval_summary = pd.read_csv(_DEFAULT_REPORTS_DIR / "evaluacion_resumen_test.csv")
    ablation = pd.read_csv(_DEFAULT_REPORTS_DIR / "ablacion_resultados.csv")

    df_test = pd.read_parquet(_DEFAULT_DATA_DIR / "03_processed" / "test_final.parquet")
    city_summary = (
        df_test.groupby("Ciudad_codigo", as_index=False)
        .agg(
            Pedidos=("ID", "count"),
            ETA_Real_Medio_min=("Time_taken (min)", "mean"),
            Dist_Media_km=("dist_haversine_km", "mean"),
            Rating_Medio=("Delivery_person_Ratings", "mean"),
            Pct_Cluster_2=("cluster_label", lambda s: float((s == 2).mean() * 100.0)),
            Pct_Trafico_Jam=(
                "Road_traffic_density",
                lambda s: float((s == 4).mean() * 100.0),
            ),
        )
        .sort_values("Pedidos", ascending=False)
        .reset_index(drop=True)
    )
    for col, digits in [
        ("ETA_Real_Medio_min", 2),
        ("Dist_Media_km", 2),
        ("Rating_Medio", 2),
        ("Pct_Cluster_2", 1),
        ("Pct_Trafico_Jam", 1),
    ]:
        city_summary[col] = city_summary[col].round(digits)

    return {
        "eval_cluster": eval_cluster,
        "eval_summary": eval_summary,
        "ablation": ablation,
        "city_summary": city_summary,
        "test_sample": df_test,
    }


def get_available_cities(imputer_artifacts: dict[str, Any] | None = None) -> list[str]:
    """Devuelve las 22 ciudades soportadas por los artefactos de entrenamiento."""
    if imputer_artifacts is None:
        imputer_artifacts = load_deployment_artifacts()["imputer_artifacts"]
    return sorted(imputer_artifacts.get("city_centroids", {}).keys())
