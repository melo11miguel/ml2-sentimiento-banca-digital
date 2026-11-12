"""Conjunto de modelado: etiquetas manuales (oro) + etiqueta débil.

Fuente de las etiquetas manuales, en orden de prioridad:
  1. data/anotacion/etiquetas_finales.csv  (tras la revisión del equipo, ver 05_kappa.py)
  2. data/anotacion/preetiquetas.csv       (provisional, preetiquetado asistido)

Reglas:
- Oro: reseñas de la muestra anotada que siguen en el corpus limpio, sin las
  marcadas como "irrelevante".
- Débil: reseñas NO anotadas con 1-2 estrellas (negativa) o 5 (positiva).
- Se excluyen las de 4 estrellas no anotadas: en la muestra aleatoria su
  etiqueta débil ("positiva") acierta en menos de un cuarto de los casos.
- Las de 3 estrellas están todas en la muestra anotada.

La columna `estrellas` se conserva solo para análisis; NUNCA es predictor,
porque de ella sale la etiqueta débil (sería fuga directa).
"""
from pathlib import Path
import sys

import pandas as pd

SIM = "--simuladas" in sys.argv   # ensayo con etiquetas ficticias
SUF = "_SIMULADO" if SIM else ""

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "processed" / "reviews_clean.csv"
ANOT = ROOT / "data" / "anotacion"
OUT = ROOT / "data" / "processed" / f"dataset{SUF}.csv"
RUIDO = ROOT / "reports" / "ruido_etiqueta_debil.csv"

DEBIL = {1: "negativa", 2: "negativa", 5: "positiva"}


def etiquetas_manuales() -> tuple[pd.DataFrame, str]:
    muestra = pd.read_csv(ANOT / "muestra_maestra.csv")
    finales = ANOT / f"etiquetas_finales{SUF}.csv"
    if finales.exists():
        et = pd.read_csv(finales)[["id", "etiqueta"]]
        return muestra.merge(et, on="id"), f"etiquetas_finales{SUF}.csv"
    pre = pd.read_csv(ANOT / "preetiquetas.csv")
    muestra["etiqueta"] = pre.sort_values("fila").preetiqueta.values
    return muestra, "preetiquetas (PROVISIONAL)"


def main():
    df = pd.read_csv(CLEAN)
    manual, origen = etiquetas_manuales()
    manual = manual[manual.id.isin(df.id)]

    # Ruido de la etiqueta débil, estimado en la parte aleatoria de la muestra.
    m = manual[manual.etiqueta != "irrelevante"].copy()
    m["debil"] = m.estrellas.map({1: "negativa", 2: "negativa", 4: "positiva", 5: "positiva"})
    ruido = (m[m.estrellas != 3].assign(acierto=lambda d: d.debil == d.etiqueta)
             .groupby("estrellas")
             .agg(n=("acierto", "size"), acierto=("acierto", "mean"),
                  conflicto=("etiqueta", lambda s: (s == "conflicto").mean())))
    RUIDO.parent.mkdir(parents=True, exist_ok=True)
    ruido.round(3).to_csv(RUIDO)

    oro = df.merge(manual[["id", "etiqueta"]], on="id")
    oro = oro[oro.etiqueta != "irrelevante"].assign(fuente="oro")
    debil = df[~df.id.isin(manual.id) & df.estrellas.isin(DEBIL)].copy()
    debil["etiqueta"] = debil.estrellas.map(DEBIL)
    debil["fuente"] = "debil"

    data = pd.concat([oro, debil], ignore_index=True)
    data = data[["id", "entidad", "texto", "fecha", "version", "votos_utiles",
                 "estrellas", "etiqueta", "fuente"]]
    data.to_csv(OUT, index=False)

    # Estimación global del ruido en la parte débil (ponderada por estrellas).
    pesos = debil.estrellas.value_counts()
    err = sum(pesos[e] * (1 - ruido.loc[e, "acierto"]) for e in pesos.index) / pesos.sum()

    print(f"Etiquetas manuales: {origen}")
    print(ruido.round(3))
    print(f"Error estimado de la etiqueta débil usada: {err:.1%}")
    print(pd.crosstab(data.fuente, data.etiqueta, margins=True))


if __name__ == "__main__":
    main()
