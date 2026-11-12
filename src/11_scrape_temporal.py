"""Conjunto de prueba temporal: reseñas publicadas DESPUÉS del corpus de modelado.

El corpus de modelado se congeló el 2 de octubre de 2026 (última reseña de cada
app el 1 de octubre). Aquí extraemos, con la misma librería, tienda e idioma,
todas las reseñas más recientes hasta llegar a esa fecha de corte. Nunca se usan
para entrenar ni para elegir hiperparámetros: solo para la evaluación final.

Uso: python src/11_scrape_temporal.py
Salida: data/raw/reviews_temporal_raw.csv
"""
from pathlib import Path
import time

import pandas as pd
from google_play_scraper import Sort, reviews

from importlib import import_module

APPS = import_module("01_scrape").APPS
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "reviews_raw.csv"
OUT = ROOT / "data" / "raw" / "reviews_temporal_raw.csv"
TOPE = 15_000   # por app, solo como salvaguarda


def extraer_hasta(app_id: str, corte: pd.Timestamp) -> list:
    filas, token = [], None
    while len(filas) < TOPE:
        lote, token = reviews(app_id, lang="es", country="co", sort=Sort.NEWEST,
                              count=200, continuation_token=token)
        if not lote:
            break
        filas.extend(lote)
        if pd.Timestamp(lote[-1]["at"]) <= corte or token is None or token.token is None:
            break
        time.sleep(0.3)
    return [f for f in filas if pd.Timestamp(f["at"]) > corte]


def main():
    cortes = pd.read_csv(RAW, parse_dates=["fecha"]).groupby("entidad").fecha.max()
    partes = []
    for entidad, app_id in APPS.items():
        df = pd.DataFrame(extraer_hasta(app_id, cortes[entidad]))
        df["entidad"] = entidad
        partes.append(df)
        print(f"{entidad}: {len(df)} reseñas posteriores a {cortes[entidad]}", flush=True)
    df = pd.concat(partes, ignore_index=True)
    df = df[["reviewId", "entidad", "content", "score", "thumbsUpCount",
             "reviewCreatedVersion", "at"]].rename(columns={
        "reviewId": "id", "content": "texto", "score": "estrellas",
        "thumbsUpCount": "votos_utiles", "reviewCreatedVersion": "version", "at": "fecha"})
    limpieza = import_module("02_clean")   # mismo anonimizado que el corpus
    df["texto"] = df.texto.map(lambda t: limpieza.anonimizar(t) if isinstance(t, str) else t)
    df.to_csv(OUT, index=False)
    print(f"Total: {len(df)} -> {OUT}")
    print(df.groupby("entidad").fecha.agg(["min", "max", "size"]))


if __name__ == "__main__":
    main()
