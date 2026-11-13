"""Exporta el modelo final a JSON para la versión web (web/, desplegada en Vercel).

Vercel no ejecuta Streamlit (no mantiene un proceso de Python encendido), así que
la versión web reimplementa en JavaScript la misma inferencia: normalización,
TF-IDF de n-gramas de caracteres (char_wb) y regresión logística multinomial.
Este script escribe:

- web/model.json: vocabulario, IDF, coeficientes, intercepto y métricas.
- tests/referencia_web.json: textos crudos con las probabilidades de sklearn,
  para que tests/check_web.mjs compruebe que la web predice lo mismo.
"""
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from comun import CLASES, SEED  # noqa: E402

# Casos difíciles para la normalización (tildes, ñ, emojis, entidades, URL, números).
CASOS = [
    "Es buena pero se cae mucho",
    "Funcionaba de maravilla, desde la actualización no abre",
    "Nequi es lo máximo 😍😍😍 pero hoy no me deja entrar",
    "DaviPlata y david plata y Davivienda, ¿cuál es cuál?",
    "Me llamo David y uso Nubank, digo, Nu bank, nuu",
    "El ñu de la sabana no es Nu; año, niño, ÑANDÚ",
    "Caidaaaaa todo el díaaaa!!! 3 veces 2024",
    "Visiten https://ejemplo.com/a?b=1 para <correo> y <numero> con <nombre>",
    "  espacios\t\tde   más \n y saltos\r\nde línea  ",
    "Pésimo servicio, llevo tres días sin poder retirar mi plata",
    "¡¡¡Excelente!!! 10/10 súper práctica ✓",
    "ok",
    "",
    "nequiiii nequí NEQUI nequis davi daviplataaa",
    "²³ número ½ y ① en la app",
    "app_mala__ muy_mala",
]


def main():
    modelo = joblib.load(ROOT / "models" / "modelo_final.joblib")
    meta = json.loads((ROOT / "models" / "metadata.json").read_text())
    tf = modelo.named_steps["tfidf"]
    lr = modelo.named_steps["clf"]
    vocab = [None] * len(tf.vocabulary_)
    for g, i in tf.vocabulary_.items():
        vocab[i] = g
    salida = {
        "clases": CLASES,
        "ngram_range": list(tf.ngram_range),
        "vocab": vocab,
        "idf": tf.idf_.tolist(),
        "coef": lr.coef_.tolist(),
        "intercept": lr.intercept_.tolist(),
        "meta": meta,
    }
    web = ROOT / "web"
    web.mkdir(exist_ok=True)
    (web / "model.json").write_text(json.dumps(salida, ensure_ascii=False, separators=(",", ":")))

    # Referencia: toda la prueba temporal más una muestra del corpus y los casos difíciles.
    pred = pd.read_csv(ROOT / "reports" / "predicciones_prueba.csv").texto.astype(str).tolist()
    corpus = pd.read_csv(ROOT / "data" / "processed" / "dataset.csv").texto.astype(str)
    muestra = corpus.sample(4000, random_state=SEED).tolist()
    textos = CASOS + pred + muestra
    p = modelo.predict_proba(textos)
    norm = modelo.named_steps["normalizar"].transform(textos)
    (ROOT / "tests").mkdir(exist_ok=True)
    (ROOT / "tests" / "referencia_web.json").write_text(json.dumps(
        {"textos": textos, "normalizados": list(norm), "prob": p.tolist()}, ensure_ascii=False))
    print(f"model.json: {len(vocab)} n-gramas; referencia: {len(textos)} textos")


if __name__ == "__main__":
    main()
