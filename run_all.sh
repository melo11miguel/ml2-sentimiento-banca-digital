#!/usr/bin/env bash
# Flujo completo, en orden. Las extracciones (01 y 11) no se repiten por defecto
# porque Google Play devuelve reseñas distintas cada día: los corpus congelados
# están en data/raw/. Para repetirlas: ./run_all.sh --extraer
set -euo pipefail
cd "$(dirname "$0")/src"
PY=../.venv/bin/python
if [[ "${1:-}" == "--extraer" ]]; then $PY 01_scrape.py; $PY 11_scrape_temporal.py; fi
$PY 02_clean.py
$PY 04_dataset.py
$PY 05_reproduccion.py
$PY 06_representacion.py
rm -rf ../reports/nested && $PY 08_modelos.py
$PY 09_interpretacion.py
$PY 10_figuras.py
$PY 12_temporal.py
$PY 13_hoja_temporal.py
$PY 14_evaluacion_temporal.py
$PY 15_interpretacion_final.py
$PY 16_exportar_web.py
