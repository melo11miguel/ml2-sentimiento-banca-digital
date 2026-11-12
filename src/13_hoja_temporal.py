"""Une las preetiquetas del conjunto temporal y genera su hoja de revisión.

Las preetiquetas (asistidas por IA, misma guía que la muestra de entrenamiento)
se hicieron leyendo SOLO el texto: sin estrellas, sin entidad y sin ver
predicciones del modelo. El equipo revisa la columna `etiqueta` en
hoja_revision_temporal.xlsx; 15_evaluacion_final.py usa esa hoja si existe la
columna revisada y, si no, las preetiquetas.

Además de la guía, marcamos como irrelevante (se descartan) las reseñas que
claramente no son de Colombia (pesos mexicanos, "chido", Cinemex) o que no
están en español, igual que el filtro de México de 02_clean.py.
"""
from pathlib import Path

import pandas as pd

ANOT = Path(__file__).resolve().parents[1] / "data" / "anotacion"
COD = {"N": "negativa", "P": "positiva", "C": "conflicto", "I": "irrelevante"}
DUDAS = {25, 77, 111, 116, 126, 197, 220, 242, 246, 253, 270, 282, 461, 488, 508, 524,
         526, 564, 593}


def main():
    etiquetas = {}
    for linea in (ANOT / "preetiquetas_temporal_parcial.txt").read_text().split("\n"):
        if linea.strip():
            ini, cods = linea.split()
            for j, c in enumerate(cods):
                etiquetas[int(ini) + j] = COD[c]
    m = pd.read_csv(ANOT / "muestra_temporal.csv")
    assert sorted(etiquetas) == list(range(len(m))), "faltan o sobran preetiquetas"
    m["preetiqueta"] = m.fila.map(etiquetas)
    m["duda_ia"] = m.fila.isin(DUDAS).map({True: "x", False: ""})
    m[["fila", "id", "preetiqueta", "duda_ia"]].to_csv(ANOT / "preetiquetas_temporal.csv",
                                                       index=False)
    hoja = m[["fila", "id", "texto", "preetiqueta", "duda_ia"]].assign(
        etiqueta=m.preetiqueta, revisado="")
    ruta = ANOT / "hoja_revision_temporal.xlsx"
    if not ruta.exists():           # nunca sobrescribir una revisión en curso
        hoja.to_excel(ruta, index=False)
    print(m.preetiqueta.value_counts())
    print(pd.crosstab(m.entidad, m.preetiqueta, margins=True))


if __name__ == "__main__":
    main()
