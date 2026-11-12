# Análisis de sentimiento en reseñas de banca digital colombiana

Proyecto integrador de Machine Learning II (Universidad Externado de Colombia).
Reproducción y adaptación de Alawaji y Aloraini (2025), *Sentiment Analysis of Digital
Banking Reviews Using Machine Learning and Large Language Models*, Electronics 14(11), 2125.

Integrantes: Miguel Ángel Melo y Nicolás Mateo Corzo.

## Instalación

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

En macOS sin Homebrew, XGBoost y LightGBM no encuentran `libomp`. Los enlazamos a la copia
que trae scikit-learn:

```bash
SP=.venv/lib/python3.9/site-packages
for f in $SP/xgboost/lib/libxgboost.dylib $SP/lightgbm/lib/lib_lightgbm.dylib; do
  install_name_tool -add_rpath @loader_path/../../sklearn/.dylibs $f && codesign --force -s - $f
done
```

## Ejecución

`./run_all.sh` reproduce todo con semilla fija (42) a partir de los corpus congelados en
`data/raw/`: `reviews_raw.csv` (modelado, extraído el 2 de octubre de 2026) y
`reviews_temporal_raw.csv` (prueba temporal, reseñas del 1 de octubre al 11 de noviembre de 2026).

### Aplicación

```bash
.venv/bin/streamlit run app/app.py
```

Abre en `http://localhost:8501`. Recibe una reseña (o un CSV con una columna `texto`),
devuelve la clase y las probabilidades, y explica la predicción con SHAP por palabra. Usa
`models/modelo_final.joblib`, que trae la normalización y el enmascaramiento de entidades
dentro del Pipeline, así que aplica exactamente el mismo preprocesamiento que en la evaluación.

| Script | Qué hace |
|---|---|
| `src/01_scrape.py` | Extrae 10 000 reseñas por app (Nequi, DaviPlata, Nu) de Google Play Colombia |
| `src/02_clean.py` | Anonimiza, filtra textos cortos, duplicados, otros idiomas y reseñas de México |
| `src/03_muestra_anotacion.py` | Muestra para anotación manual (se ejecutó una sola vez; no repetir) |
| `src/hojas_anotacion.py`, `src/07_kappa.py` | Hojas de anotación, kappa de Cohen y etiquetas finales |
| `src/04_dataset.py` | Une etiquetas manuales (oro) y etiqueta débil por estrellas |
| `src/comun.py` | Normalización, Pipeline, validación cruzada y métricas compartidas |
| `src/05_reproduccion.py` | Experimento 1: protocolo del artículo tal cual |
| `src/06_representacion.py` | Experimento 2: representación del texto × tratamiento del desbalance |
| `src/08_modelos.py` | Experimento 3: once modelos con validación cruzada anidada |
| `src/09_interpretacion.py` | Experimento 4 (ablación) e interpretación preliminar |
| `src/10_figuras.py` | Figuras de la Entrega 2 |
| `src/11_scrape_temporal.py` | Extrae las reseñas publicadas después del corpus (prueba temporal) |
| `src/12_temporal.py` | Depura el conjunto temporal con las mismas reglas y saca la muestra de 600 |
| `src/13_hoja_temporal.py` | Preetiquetas y hoja de revisión del conjunto temporal |
| `src/14_evaluacion_temporal.py` | Validación y prueba temporal, modelo final y su versión de despliegue |
| `src/15_interpretacion_final.py` | SHAP global y local, y figuras de la entrega final |
| `src/explicar.py` | Explicaciones SHAP por palabra (las usa también la app) |
| `app/app.py` | Despliegue en Streamlit |

## Etiquetado

Ver `data/anotacion/GUIA_ANOTACION.md`. Las dos muestras anotadas, la de modelado (1 440
reseñas) y la temporal (600), están **preetiquetadas con asistencia de IA** siguiendo esa
guía, leyendo solo el texto (sin estrellas ni predicciones). La revisión humana se hace en
`hoja_revision.xlsx` y `hoja_revision_temporal.xlsx`, y el acuerdo entre anotadores (kappa de
Cohen) se calcula con `src/07_kappa.py` a partir de las hojas ciegas (120 reseñas). Mientras
esas hojas no estén anotadas, todas las cifras usan las preetiquetas y el kappa no existe.

## Uso de IA generativa

Usamos Claude (Anthropic) como apoyo en la escritura de código, el preetiquetado de la
muestra y la revisión de redacción. Todas las decisiones fueron revisadas por el equipo.
