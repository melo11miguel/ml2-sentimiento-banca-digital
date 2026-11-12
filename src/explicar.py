"""Explicaciones SHAP por palabra, compartidas por el análisis y la app.

El modelo final usa n-gramas de caracteres, que no se leen. Por eso explicamos
con SHAP sobre el TEXTO: el explicador de partición quita palabras completas
(máscara de texto) y mide cuánto cambia la probabilidad de cada clase. Los
aportes de las palabras más la probabilidad base suman exactamente la
probabilidad predicha (propiedad de aditividad de SHAP).
"""
from pathlib import Path
import sys

import numpy as np
import shap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import CLASES  # noqa: E402


def explicador(modelo, max_evals=300):
    masker = shap.maskers.Text(r"\W+")
    f = lambda textos: modelo.predict_proba([str(t) for t in textos])  # noqa: E731
    ex = shap.Explainer(f, masker, output_names=CLASES)
    return lambda textos: ex(list(textos), max_evals=max_evals, silent=True)


def palabras(sv, i):
    """Lista de (palabra, aportes[3]) de la reseña i, sin espacios sobrantes."""
    return [(str(tok).strip(), np.asarray(v)) for tok, v in zip(sv.data[i], sv.values[i])
            if str(tok).strip()]
