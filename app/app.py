"""Despliegue básico del modelo final.

Ejecutar desde la raíz del repositorio:
    .venv/bin/streamlit run app/app.py

La app carga models/modelo_final.joblib, que incluye la normalización del texto
y el enmascaramiento de entidades DENTRO del Pipeline: recibe la reseña tal
como la escribe el usuario, igual que en la prueba temporal.
"""
from html import escape
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from comun import CLASES  # noqa: E402  (también lo necesita el joblib)
from explicar import explicador, palabras  # noqa: E402

COLOR = {"negativa": "#2a78d6", "positiva": "#eb6834", "conflicto": "#1baf7a"}
EJEMPLOS = [
    "Excelente app, muy fácil de usar, pero desde la última actualización no me deja entrar",
    "Pésimo servicio, llevo tres días sin poder retirar mi plata",
    "Me encanta, las transferencias llegan al instante y la tarjeta es gratis",
    "Es buena pero deberían quitar el cobro del 4x1000",
]

st.set_page_config(page_title="Sentimiento en banca digital", page_icon="🏦", layout="centered")


@st.cache_resource
def cargar():
    modelo = joblib.load(ROOT / "models" / "modelo_final.joblib")
    meta = json.loads((ROOT / "models" / "metadata.json").read_text())
    return modelo, meta, explicador(modelo)


def resaltar(pares, k):
    """HTML con cada palabra coloreada según su aporte SHAP a la clase k."""
    tope = max(0.05, max(abs(v[k]) for _, v in pares))
    partes = []
    for tok, v in pares:
        a = float(v[k]) / tope
        fondo = (f"rgba(27,175,122,{min(abs(a), 1):.2f})" if a > 0
                 else f"rgba(190,60,60,{min(abs(a), 1):.2f})")
        partes.append(f'<span title="{v[k]:+.3f}" style="background:{fondo};'
                      f'padding:1px 3px;border-radius:3px;margin:1px">{escape(tok)}</span>')
    return " ".join(partes)


modelo, meta, explicar = cargar()
pt = meta["prueba_temporal"]

with st.sidebar:
    st.header("Sobre el modelo")
    n_ent = f"{meta['n_entrenamiento']:,}".replace(",", ".")
    st.markdown(
        f"Regresión logística sobre n-gramas de caracteres, con ponderación de clases. "
        f"Entrenada con {n_ent} reseñas de Nequi, DaviPlata y Nu "
        f"(Google Play Colombia), variante: *{meta['entrenamiento']}*.")
    st.metric("F1 macro en prueba temporal", f"{pt['f1_macro']:.2f}".replace(".", ","))
    st.metric("F1 de conflicto", f"{pt['f1_conflicto']:.2f}".replace(".", ","))
    st.caption(
        f"Prueba temporal: {pt['n']} reseñas publicadas entre el 30 de octubre y el 11 de "
        "noviembre de 2026, nunca vistas en el entrenamiento. Los nombres de las entidades "
        "se enmascaran antes de predecir. Es un instrumento de monitoreo, no un clasificador "
        "autónomo: conviene revisar las reseñas marcadas como conflicto.")

st.title("Sentimiento en reseñas de banca digital")
st.write("Clasifica una reseña como **negativa**, **positiva** o **conflicto** "
         "(elogio y queja en la misma reseña) y explica qué palabras pesaron.")

tab1, tab2 = st.tabs(["Una reseña", "Varias reseñas (CSV)"])

with tab1:
    ejemplo = st.selectbox("Ejemplos", ["(escribir la mía)"] + EJEMPLOS)
    texto = st.text_area("Reseña", value="" if ejemplo.startswith("(") else ejemplo,
                         height=110, placeholder="Escribe aquí la reseña...")
    if st.button("Clasificar", type="primary"):
        if len(texto.split()) == 0:
            st.warning("Escribe una reseña.")
        else:
            if len(texto.split()) < 3:
                st.info("El modelo se entrenó con reseñas de tres palabras o más; "
                        "con textos tan cortos la predicción es poco confiable.")
            prob = modelo.predict_proba([texto])[0]
            k = int(np.argmax(prob))
            clase = CLASES[k]
            st.markdown(f"### Predicción: <span style='color:{COLOR[clase]}'>{clase}</span>",
                        unsafe_allow_html=True)
            st.bar_chart(pd.DataFrame({"probabilidad": prob}, index=CLASES),
                         horizontal=True, height=160)
            with st.spinner("Calculando la explicación SHAP..."):
                pares = palabras(explicar([texto]), 0)
            st.markdown(f"**Por qué**: en verde, palabras que suben la probabilidad de "
                        f"*{clase}*; en rojo, las que la bajan.")
            st.markdown(resaltar(pares, k), unsafe_allow_html=True)
            tabla = pd.DataFrame([{"palabra": t} | {c: float(v[j]) for j, c in
                                                    enumerate(CLASES)} for t, v in pares])
            with st.expander("Aportes SHAP por palabra y clase"):
                st.dataframe(tabla.round(3), hide_index=True)

with tab2:
    st.write("Sube un CSV con una columna `texto`. Devuelve la clase y las probabilidades.")
    archivo = st.file_uploader("Archivo CSV", type="csv")
    if archivo is not None:
        df = pd.read_csv(archivo)
        if "texto" not in df.columns:
            st.error("El archivo no tiene una columna llamada `texto`.")
        else:
            textos = df.texto.fillna("").astype(str).tolist()
            p = modelo.predict_proba(textos)
            df["prediccion"] = [CLASES[i] for i in p.argmax(1)]
            for j, c in enumerate(CLASES):
                df[f"p_{c}"] = p[:, j].round(3)
            st.dataframe(df, hide_index=True)
            st.bar_chart(df.prediccion.value_counts())
            st.download_button("Descargar resultados", df.to_csv(index=False).encode("utf-8"),
                               "predicciones.csv", "text/csv")
