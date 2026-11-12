"""Depuración del conjunto temporal y muestra para anotación.

1. Mismas reglas de 02_clean.py (función `depurar`), sin cambiar nada.
2. Se quitan los textos que ya estaban en el corpus de modelado (misma clave
   normalizada): un texto repetido no sería una reseña "nunca vista".
3. Muestra de N_POR_APP reseñas por entidad, al azar y SIN estratificar por
   estrellas: conserva la distribución natural de clases, que es el escenario
   real de uso del modelo.

Uso: python src/12_temporal.py
Salidas: data/processed/temporal_clean.csv, data/anotacion/muestra_temporal.csv,
         data/anotacion/lotes_temporal/*.txt (solo índice y texto, sin estrellas),
         reports/embudo_temporal.csv
"""
from importlib import import_module
from pathlib import Path

import pandas as pd

limpieza = import_module("02_clean")
SEED = 42
N_POR_APP = 200
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "reviews_temporal_raw.csv"
CLEAN = ROOT / "data" / "processed" / "temporal_clean.csv"
CORPUS = ROOT / "data" / "processed" / "reviews_clean.csv"
ANOT = ROOT / "data" / "anotacion"


def main():
    df, embudo = limpieza.depurar(pd.read_csv(RAW))
    vistos = set(pd.read_csv(CORPUS).texto.map(limpieza.normalizar))
    df = df[~df.texto.map(limpieza.normalizar).isin(vistos)]
    embudo.append(("no vistas en el corpus", len(df)))
    df.to_csv(CLEAN, index=False)
    pd.DataFrame(embudo, columns=["paso", "n"]).to_csv(
        ROOT / "reports" / "embudo_temporal.csv", index=False)
    for paso, n in embudo:
        print(f"{paso:>24}: {n}")

    muestra = (df.groupby("entidad", group_keys=False)
               .apply(lambda g: g.sample(N_POR_APP, random_state=SEED))
               .sample(frac=1, random_state=SEED).reset_index(drop=True))
    muestra = muestra[["id", "entidad", "estrellas", "fecha", "version", "texto"]]
    muestra.to_csv(ANOT / "muestra_temporal.csv", index_label="fila")
    print(pd.crosstab(muestra.entidad, muestra.estrellas, margins=True))

    lotes = ANOT / "lotes_temporal"
    lotes.mkdir(exist_ok=True)
    for i in range(0, len(muestra), 100):
        txt = "\n".join(f"{j}\t{t}" for j, t in muestra.texto.iloc[i:i + 100].items())
        (lotes / f"t{i // 100:02d}.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
