"""Muestra para anotación manual.

- Todas las reseñas de 3 estrellas (candidatas naturales a conflicto).
- N_OTRAS reseñas aleatorias de cada nivel 1, 2, 4 y 5, para estimar la tasa
  de error de la etiqueta débil y cuánto conflicto queda escondido en ellas.
- Un subconjunto ciego de N_CIEGA reseñas que dos anotadores etiquetan por
  separado, sin ver estrellas ni preetiqueta, para calcular kappa de Cohen.

Las estrellas NO se muestran en las hojas de anotación para no sesgar al
anotador hacia la etiqueta débil.
"""
from pathlib import Path

import pandas as pd

SEED = 42
N_OTRAS = 75
N_CIEGA = 200

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "processed" / "reviews_clean.csv"
DIR = ROOT / "data" / "anotacion"


def main():
    df = pd.read_csv(CLEAN)
    tres = df[df.estrellas == 3]
    otras = df[df.estrellas != 3].groupby("estrellas").sample(N_OTRAS, random_state=SEED)
    muestra = pd.concat([tres, otras]).sample(frac=1, random_state=SEED)
    muestra = muestra[["id", "entidad", "estrellas", "texto"]].reset_index(drop=True)
    muestra["ciega"] = False
    ciega_idx = (muestra.groupby("estrellas")
                 .sample(frac=N_CIEGA / len(muestra), random_state=SEED).index)
    muestra.loc[ciega_idx, "ciega"] = True

    DIR.mkdir(parents=True, exist_ok=True)
    muestra.to_csv(DIR / "muestra_maestra.csv", index=False)
    print(muestra.groupby(["estrellas", "ciega"]).size().unstack())


if __name__ == "__main__":
    main()
