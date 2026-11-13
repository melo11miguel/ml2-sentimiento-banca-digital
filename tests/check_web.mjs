// Paridad de la versión web con sklearn: misma normalización, mismas probabilidades
// y misma clase en todos los textos de tests/referencia_web.json (generado por
// src/16_exportar_web.py). Uso: node tests/check_web.mjs
import { readFileSync } from "node:fs";
import { Modelo, normalizar } from "../web/modelo.js";

const raiz = new URL("..", import.meta.url);
const modelo = new Modelo(JSON.parse(readFileSync(new URL("web/model.json", raiz))));
const ref = JSON.parse(readFileSync(new URL("tests/referencia_web.json", raiz)));

let difNorm = 0, difClase = 0, maxDif = 0;
ref.textos.forEach((t, i) => {
  const n = normalizar(t);
  if (n !== ref.normalizados[i]) {
    difNorm++;
    if (difNorm <= 5) console.log("normalización distinta:", JSON.stringify(t), "\n  js:", JSON.stringify(n), "\n  py:", JSON.stringify(ref.normalizados[i]));
  }
  const p = modelo.proba(t);
  const q = ref.prob[i];
  maxDif = Math.max(maxDif, ...p.map((v, k) => Math.abs(v - q[k])));
  if (p.indexOf(Math.max(...p)) !== q.indexOf(Math.max(...q))) difClase++;
});

// Aditividad de la explicación: base + aportes = predicción.
let maxAdit = 0;
for (const t of ["Es buena pero se cae mucho", ref.textos[20], ref.textos[400]]) {
  const e = modelo.explicar(t);
  const p = modelo.proba(t);
  for (let k = 0; k < 3; k++) {
    const s = e.base[k] + e.palabras.reduce((a, w) => a + w.aporte[k], 0);
    maxAdit = Math.max(maxAdit, Math.abs(s - p[k]));
  }
}

console.log(`textos: ${ref.textos.length}`);
console.log(`normalización distinta: ${difNorm}`);
console.log(`clase distinta: ${difClase}`);
console.log(`diferencia máxima de probabilidad: ${maxDif.toExponential(2)}`);
console.log(`error máximo de aditividad SHAP: ${maxAdit.toExponential(2)}`);
const ok = difNorm === 0 && difClase === 0 && maxDif < 1e-9 && maxAdit < 1e-9;
console.log(ok ? "OK: la web reproduce el modelo de Python" : "FALLO");
process.exit(ok ? 0 : 1);
