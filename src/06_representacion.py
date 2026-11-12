"""Experimento 2: representación del texto y tratamiento del desbalance.

Diseño factorial con dos clasificadores lineales (rápidos y estables en
espacios TF-IDF dispersos) y parámetros por defecto, para aislar el efecto de:
- Representación: artículo (unigramas sin palabras vacías NLTK), unigramas,
  uni+bigramas, n-gramas de caracteres (2-5) y palabras+caracteres.
- Desbalance: ninguno, ponderación de clases, SMOTE (dentro del Pipeline,
  así que solo se sobremuestrea el pliegue de entrenamiento).

Salida: reports/exp2_representacion.csv
"""
import itertools
import warnings

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVC

from comun import REPORTS, SCORERS, SEED, cargar, pipeline, resumir, splits

warnings.filterwarnings("ignore")

REPS = ["articulo", "uni", "uni_bi", "char", "palabra_char"]
BALANCE = ["ninguno", "ponderacion", "smote"]


def clf(nombre, balance):
    cw = "balanced" if balance == "ponderacion" else None
    if nombre == "LR":
        return LogisticRegression(class_weight=cw, max_iter=3000)
    return LinearSVC(class_weight=cw, random_state=SEED)


def main():
    _, X, y = cargar()
    sp = splits(X, y)
    filas = []
    for modelo, rep, bal in itertools.product(["LR", "SVM"], REPS, BALANCE):
        pipe = pipeline(clf(modelo, bal), rep=rep, smote=(bal == "smote"))
        res = cross_validate(pipe, X, y, cv=sp, scoring=SCORERS, n_jobs=4)
        fila = resumir(modelo, res) | {"representacion": rep, "balance": bal}
        filas.append(fila)
        print(f"{modelo:3s} {rep:13s} {bal:12s} F1m_oro={fila['f1_macro_oro']:.3f}"
              f"±{fila['f1_macro_oro_sd']:.3f} F1conf={fila['f1_conflicto_oro']:.3f}",
              flush=True)
    pd.DataFrame(filas).to_csv(REPORTS / "exp2_representacion.csv", index=False)


if __name__ == "__main__":
    main()
