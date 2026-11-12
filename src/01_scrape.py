"""Extracción de reseñas de Google Play (tienda Colombia, idioma español).

Replica el procedimiento del artículo base (google-play-scraper) sobre tres
apps de banca digital colombiana. Se toman las reseñas más recientes para
conservar la distribución natural de calificaciones (no se estratifica por
estrellas en la extracción).

Uso: python src/01_scrape.py
Salida: data/raw/reviews_raw.csv
"""
from importlib import import_module
from pathlib import Path
import time

import pandas as pd
from google_play_scraper import Sort, reviews

APPS = {
    "Nequi": "com.nequi.MobileApp",
    "DaviPlata": "com.davivienda.daviplataapp",
    "Nu": "com.nu.production",
}
N_POR_APP = 10_000
LOTE = 200

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "reviews_raw.csv"


def extraer(app_id: str, n: int) -> list:
    filas, token = [], None
    while len(filas) < n:
        lote, token = reviews(
            app_id, lang="es", country="co", sort=Sort.NEWEST,
            count=min(LOTE, n - len(filas)), continuation_token=token,
        )
        if not lote:
            break
        filas.extend(lote)
        time.sleep(0.3)
        if token is None or token.token is None:
            break
    return filas


def main():
    partes = []
    for entidad, app_id in APPS.items():
        filas = extraer(app_id, N_POR_APP)
        df = pd.DataFrame(filas)
        df["entidad"] = entidad
        partes.append(df)
        print(f"{entidad}: {len(df)} reseñas")
    df = pd.concat(partes, ignore_index=True)
    cols = ["reviewId", "entidad", "content", "score", "thumbsUpCount",
            "reviewCreatedVersion", "at"]
    df = df[cols].rename(columns={
        "reviewId": "id", "content": "texto", "score": "estrellas",
        "thumbsUpCount": "votos_utiles", "reviewCreatedVersion": "version",
        "at": "fecha",
    })
    # Anonimización: no se guardan userName, userImage ni respuestas del banco, y el
    # texto se anonimiza (correos, teléfonos, cédulas, nombres) antes de guardarlo.
    limpieza = import_module("02_clean")
    df["texto"] = df.texto.map(lambda t: limpieza.anonimizar(t) if isinstance(t, str) else t)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Total: {len(df)} -> {OUT}")


if __name__ == "__main__":
    main()
